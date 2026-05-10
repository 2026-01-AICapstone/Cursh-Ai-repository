import faulthandler
faulthandler.enable()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
from dataclasses import dataclass

import numpy as np
import torch
import transformers
# import wandb

# M1에서 호환되지 않는 DeepSpeed 임포트 주석 처리
# from deepspeed import zero
# from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from args import (HyperparamArguments, LoraArguments, ModelArguments,
                  TrainingArguments)
from classifier import HarmbenchClassifier
from dataset import RepBendingDataset
from trainer import CustomTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 🔥 [CRUSH] crush_loss.py에서 CentroidManager 불러오기
from crush_loss import CentroidManager


def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k, []).append(input)

    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.cat(inputs, dim=0)
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        else:
            batch_inputs[k] = inputs
    return batch_inputs


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
    )
    (
        model_args,
        training_args,
        lora_args,
        hyperparam_args,
    ) = parser.parse_args_into_dataclasses()

    print(hyperparam_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)

    device_map = "auto"
    model_name_or_path = model_args.model_name_or_path

    # 🔥 [CRUSH] 모델의 config를 먼저 불러와 전체 레이어 수(Architecture) 파악
    config = AutoConfig.from_pretrained(model_name_or_path)
    total_layers = config.num_hidden_layers

    target_layers = hyperparam_args.target_layers
    target_layer_start_idx = hyperparam_args.target_layer_start_idx
    layers_window_size = hyperparam_args.layers_window_size
    transform_layers = hyperparam_args.transform_layers
    full_layers = hyperparam_args.full_layers

    # 🔥 [CRUSH] Qwen, LLaMA 등 모델에 따른 중후반 레이어 자동 감지 로직
    auto_start_layer = int(total_layers * 0.5)
    auto_end_layer = total_layers - 1

    if target_layers == "" or target_layers == "-1":
        # 쉘 스크립트에서 "-1"을 넘겼을 경우 자동 계산 (예: 32레이어면 16~30)
        hyperparam_args.target_layers = list(range(auto_start_layer, auto_end_layer))
        print(f"✅ 모델 자동 감지: 총 {total_layers}개 레이어 중 {auto_start_layer}번째부터 {auto_end_layer - 1}번째 레이어를 타겟으로 설정합니다.")
    elif layers_window_size > 0 and target_layers != "-1":
        hyperparam_args.target_layers = list(range(target_layer_start_idx, target_layer_start_idx + layers_window_size))
    else:
        hyperparam_args.target_layers = [int(layer) for layer in target_layers.split(",")]

    # LoRA 변환 레이어도 타겟 레이어와 동일하게 설정
    if "-1" in transform_layers:
        lora_layers_to_transform = hyperparam_args.target_layers
    else:
        lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )

    drop_layers_after = max(hyperparam_args.target_layers) if not full_layers else None
    print("lora_transform_layers", lora_config.layers_to_transform)
    print("drop_layers_after", drop_layers_after)

    if drop_layers_after:
        config.num_hidden_layers = drop_layers_after + 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=training_args.max_seq_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_dataset = RepBendingDataset(tokenizer, num_examples=500, mode=hyperparam_args.loss_mode, max_length=512,
                                      model_name_or_path=model_name_or_path, dataset_path=hyperparam_args.dataset_path,
                                      split=hyperparam_args.dataset_split, is_online=hyperparam_args.is_online)

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4bit 양자화
        bnb_4bit_quant_type="nf4",  # 양자화 방식 (nf4 권장)
        bnb_4bit_compute_dtype=torch.float16,  # 연산은 fp16으로
        bnb_4bit_use_double_quant=True,  # 이중 양자화로 추가 절약
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    training_args.model_name_or_path = model_name_or_path

    model = get_peft_model(model, lora_config)
    print("model", model)

    if getattr(training_args, "deepspeed", None) is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    print("TRAIN SIZE: ", len(train_dataset))

    if hyperparam_args.is_online:
        classifier = HarmbenchClassifier()
    else:
        classifier = None

    training_args.remove_unused_columns = False

    # 🔥 [CRUSH] Seeded K-Means로 분류한 클러스터 수(5개)로 세팅
    NUM_CATEGORIES = 5
    centroid_mgr = CentroidManager(num_classes=NUM_CATEGORIES, hidden_size=model.config.hidden_size)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        max_seq_length=training_args.max_seq_length,
        packing=True,
        hyperparam_args=hyperparam_args,
        classifier=classifier,
        centroid_mgr=centroid_mgr  # 세팅된 관리자를 CustomTrainer에 전달
    )
    model.config.use_cache = False

    trainer.train()

    # 학습 완료 후 모델 가중치 및 최종 중심점(Centroids) 좌표 파일 저장
    trainer.save_model(training_args.output_dir)
    torch.save(centroid_mgr.centroids, f"{training_args.output_dir}/crush_centroids.pt")
    print(f"✅ 학습 완료! 중심점 데이터가 {training_args.output_dir}/crush_centroids.pt 에 저장되었습니다.")


if __name__ == "__main__":
    SEED = 42

    # 하드코딩된 CUDA 시드 설정을 디바이스(CUDA/MPS/CPU) 환경에 맞춰 동적으로 작동하게 변경
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    #torch.use_deterministic_algorithms(True)

    train()