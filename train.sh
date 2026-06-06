#!/bin/bash

# ==============================================================================
# 🌟 CRUSH 프로젝트 (잠재공간 클러스터링 기반 LLM 세밀한 유해성 탐지 및 방어)
# ==============================================================================
export MASTER_PORT=$((29000 + RANDOM % 1000))
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONUTF8=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
# -----------------------------------------------------------
# [1] 실행 환경 및 모델 선택
# -----------------------------------------------------------
ENVIRONMENT="GPU"      # "GPU" (실제 학습용) 또는 "MAC" (로컬 코드 테스트용) 선택
MODEL="Qwen_3B"      # 사용할 모델 선택: "Llama3_8B", "Qwen_3B", "Qwen_0.5B"

if [ $MODEL == "Llama3_8B" ]; then
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
elif [ $MODEL == "Qwen_3B" ]; then
    model_name_or_path="Qwen/Qwen2.5-3B-Instruct"
elif [ $MODEL == "Qwen_0.5B" ]; then
    model_name_or_path="Qwen/Qwen2-0.5B-Instruct"
fi

# -----------------------------------------------------------
# [2] CRUSH 하이퍼파라미터 설정 (기존 rep_bending을 대체)
# -----------------------------------------------------------
METHOD="Capstone" # 폴더 구조상 경로는 rep_bending 유지

# 🔥 [CRUSH 핵심 가중치 설정]
# (train.py와 trainer.py에서 설정한 CRUSH Loss 가중치로 들어갑니다)
learning_rate=1e-4
alpha=1.0   # Benign Loss 가중치 (안전한 응답 보존력)
beta=1.0    # Pull Loss 가중치 (같은 카테고리끼리 뭉치는 힘)
gamma=1.0   # Push Loss 가중치 (다른 카테고리끼리 밀어내는 힘)
epsilon=0.5 # KL Divergence 가중치 (일반 지식 유지력)
eta=0.0

# 레이어 및 학습 설정
target_layers="-1"
target_layer_start_idx=20
layers_window_size=11
transform_layers="-1"
loss_mode="response_all"
alpha_mode="all"

# 🔥 [CRUSH 핵심 데이터셋] 다중 카테고리가 포함된 데이터셋 사용
dataset_path="LibrAI/do-not-answer"
dataset_split="train"

# -----------------------------------------------------------
# [3] 환경에 따른 동적 설정 (GPU vs MAC)
# -----------------------------------------------------------
if [ $ENVIRONMENT == "GPU" ]; then
    echo "🚀 GPU 서버 모드로 실행합니다."
    export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.5 # GPU 서버 경로에 맞게 수정 필요
    export CUBLAS_WORKSPACE_CONFIG=:16:8

    max_step=500
    eval_steps=100
    max_seq_length=512
    per_device_train_batch_size=1
    gradient_accumulation_steps=8

    # GPU용 Accelerate 세팅
    ACCELERATE_ARGS=""
    DTYPE_ARGS="--bf16 False --tf32 True --fp16 True"

elif [ $ENVIRONMENT == "MAC" ]; then
    echo "💻 M1/M2 Mac 디버깅 모드로 실행합니다 (OOM 방지 설정)."
    export PYTORCH_ENABLE_MPS_FALLBACK=1

    # 메모리가 작은 Mac을 위한 초소형 세팅
    max_step=5
    eval_steps=5
    max_seq_length=256
    per_device_train_batch_size=1
    gradient_accumulation_steps=1
    target_layer_start_idx=10
    layers_window_size=5

    # Mac용 Accelerate 세팅 (DeepSpeed 끄기)
    ACCELERATE_ARGS=""
    DTYPE_ARGS="--bf16 False --tf32 False --fp16 False"
fi

# -----------------------------------------------------------
# [4] 출력 폴더명 생성 및 실행
# -----------------------------------------------------------
output="${MODEL}_CRUSH_lr${learning_rate}_a${alpha}_b${beta}_g${gamma}_eps${epsilon}_${max_step}step"
if [ $ENVIRONMENT == "MAC" ]; then
    output="${output}_mac_debug"
fi

output_dir="./out/${output}"

echo "====================================="
echo "🛠️ Model       : $model_name_or_path"
echo "📁 Dataset     : $dataset_path"
echo "💾 Output Dir  : $output_dir"
echo "====================================="

# Accelerate 실행 (환경에 따라 자동으로 인자가 바뀜)
accelerate launch \
    --num_processes 1 \
    --main_process_port $MASTER_PORT \
    $ACCELERATE_ARGS \
    train.py\
    --model_name_or_path $model_name_or_path \
    --dataset_path $dataset_path \
    --dataset_split $dataset_split \
    --target_layers $target_layers \
    --target_layer_start_idx $target_layer_start_idx \
    --layers_window_size $layers_window_size \
    --transform_layers $transform_layers \
    --loss_alpha $alpha \
    --loss_beta $beta \
    --loss_gamma $gamma \
    --loss_epsilon $epsilon \
    --loss_eta $eta \
    --loss_mode $loss_mode \
    --alpha_mode $alpha_mode \
    --max_steps $max_step \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    $DTYPE_ARGS \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --do_eval \
    --eval_strategy "steps" \
    --eval_steps $eval_steps \
    --save_total_limit 1 \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --max_seq_length $max_seq_length \
    --q_lora False\
    --gradient_checkpointing False \
    --report_to none \
    --is_online False

# (평가 쉘 스크립트 실행 부분은 GPU 서버에서 사용할 때 주석 해제하여 사용하세요)