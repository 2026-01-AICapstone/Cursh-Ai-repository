# import random
# from typing import Dict
# import jsonlines
# import torch
# import transformers
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
#
# random.seed(0)
#
#
# class RepBendingDataset(Dataset):
#
#     def __init__(self,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  num_examples,
#                  mode,
#                  max_length,
#                  model_name_or_path,
#                  dataset_path,
#                  split=None,
#                  is_online=False,
#                  ):
#         super().__init__()
#
#         self.model_name_or_path = model_name_or_path.lower()
#         self.max_length = max_length
#         self.tokenizer = tokenizer
#         self.num_examples = num_examples
#
#         self.data_safe_samples = []
#         self.data_unsafe_samples = []
#         self.data_unsafe_labels = []
#         self.retain_set = []
#         self.unsafe_prompt_pair_unsafe_answer = []
#         self.unsafe_prompt_pair_safe_answer = []
#
#         # =========================================================
#         # TEMPLATE
#         # =========================================================
#         def set_tags_templates():
#             one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
#
#             user_tag, assistant_tag = None, None
#
#             if "qwen" in self.model_name_or_path:
#                 print("USING QWEN TEMPLATE")
#                 user_tag = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
#                 assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
#             elif "llama" in self.model_name_or_path:
#                 print("USING LLAMA TEMPLATE")
#                 user_tag = "<|user|>\n"
#                 assistant_tag = "<|assistant|>\n"
#             else:
#                 raise NotImplementedError()
#
#             self.user_tag = user_tag
#             self.assistant_tag = assistant_tag
#             return one_shot_template
#
#         # =========================================================
#         # ULTRACHAT (SAFE FIXED VERSION)
#         # =========================================================
#
#         # def sample_from_ultrachat():
#         #     print("Loading UltraChat (non-streaming mode)...")
#         #
#         #     ds = load_dataset(
#         #         "HuggingFaceH4/ultrachat_200k",
#         #         split=f"train_sft[:{self.num_examples * 3}]"  # 여유분 확보
#         #     )
#         #
#         #     safe_samples = []
#         #     retain = []
#         #
#         #     for example in ds:
#         #         if len(safe_samples) >= self.num_examples:
#         #             break
#         #
#         #         messages = example["messages"]
#         #         if len(messages) < 2:
#         #             continue
#         #
#         #         instruction = messages[0]["content"]
#         #         response = messages[1]["content"]
#         #
#         #         safe_samples.append(
#         #             self.one_shot_template.format(
#         #                 user_tag=self.user_tag,
#         #                 assistant_tag=self.assistant_tag,
#         #                 instruction=instruction,
#         #                 response=response
#         #             )
#         #         )
#         #         retain.append(instruction)
#         #
#         #     self.data_safe_samples = safe_samples
#         #     self.retain_set = retain
#         #     print(f"UltraChat loaded: {len(safe_samples)}")
#
#         def sample_from_ultrachat():
#             print("Loading safe samples via HuggingFace API...")
#             import requests
#
#             url = "https://datasets-server.huggingface.co/rows"
#             safe_samples = []
#             retain = []
#             batch_size = 100  # API 최대 허용치
#
#             offset = 0
#             while len(safe_samples) < self.num_examples:
#                 params = {
#                     "dataset": "HuggingFaceH4/ultrachat_200k",
#                     "config": "default",
#                     "split": "train_sft",
#                     "offset": offset,
#                     "length": batch_size
#                 }
#
#                 response = requests.get(url, params=params, timeout=60)
#                 data = response.json()
#                 rows = data.get("rows", [])
#
#                 if not rows:
#                     break
#
#                 for row in rows:
#                     messages = row.get("row", {}).get("messages", [])
#                     if len(messages) < 2:
#                         continue
#
#                     instruction = messages[0].get("content", "")
#                     response_text = messages[1].get("content", "")
#
#                     if not instruction or not response_text:
#                         continue
#
#                     safe_samples.append(
#                         self.one_shot_template.format(
#                             user_tag=self.user_tag,
#                             assistant_tag=self.assistant_tag,
#                             instruction=instruction,
#                             response=response_text
#                         )
#                     )
#                     retain.append(instruction)
#
#                     if len(safe_samples) >= self.num_examples:
#                         break
#
#                 offset += batch_size
#                 print(f"  {len(safe_samples)}/{self.num_examples} 로드 중...")
#
#             self.data_safe_samples = safe_samples
#             self.retain_set = retain
#             print(f"Safe samples loaded: {len(safe_samples)}")
#
#
#
#         # =========================================================
#         # DO NOT ANSWER (SAFE LIMITED)
#         # =========================================================
#         def sample_from_do_not_answer_with_kmeans():
#             print("Loading Do-Not-Answer via HuggingFace API...")
#             import requests
#             from sklearn.cluster import KMeans
#             from sklearn.feature_extraction.text import TfidfVectorizer
#
#             url = "https://datasets-server.huggingface.co/rows"
#             raw_unsafe = []
#             raw_inst = []
#             seed_dict = {}
#
#             offset = 0
#             batch_size = 100
#
#             while len(raw_unsafe) < 500:
#                 params = {
#                     "dataset": "LibrAI/do-not-answer",
#                     "config": "default",
#                     "split": "train",
#                     "offset": offset,
#                     "length": batch_size
#                 }
#
#                 response = requests.get(url, params=params, timeout=60)
#                 data = response.json()
#                 rows = data.get("rows", [])
#
#                 if not rows:
#                     break
#
#                 for row in rows:
#                     r = row.get("row", {})
#                     instruction = r.get("question", r.get("prompt", ""))
#                     response_text = r.get("response", "")
#                     category = r.get("risk_area", "Unknown")
#
#                     if not instruction:
#                         continue
#
#                     formatted = self.one_shot_template.format(
#                         user_tag=self.user_tag,
#                         assistant_tag=self.assistant_tag,
#                         instruction=instruction,
#                         response=response_text
#                     )
#
#                     raw_unsafe.append(formatted)
#                     raw_inst.append(instruction)
#
#                     if category not in seed_dict:
#                         seed_dict[category] = instruction
#
#                     if len(raw_unsafe) >= 500:
#                         break
#
#                 offset += batch_size
#                 print(f"  Unsafe {len(raw_unsafe)}/500 로드 중...")
#
#             # KMeans 클러스터링
#             vectorizer = TfidfVectorizer(max_features=1000)
#             X = vectorizer.fit_transform(raw_inst)
#
#             pseudo_labels = KMeans(
#                 n_clusters=5,
#                 random_state=42,
#                 n_init=1
#             ).fit_predict(X)
#
#             self.data_unsafe_samples = raw_unsafe
#             self.data_unsafe_labels = pseudo_labels.tolist()
#
#             print(f"Unsafe loaded: {len(raw_unsafe)}")
#
#         # =========================================================
#         # WILDJAILBREAK (SAFE LIMITED)
#         # =========================================================
#         def sample_from_wildjailbreak():
#             print("Loading WildJailbreak (safe 500)...")
#
#             train_data = []
#
#             try:
#                 with jsonlines.open("./wildjailbreak.jsonl") as f:
#                     for i, line in enumerate(f):
#                         train_data.append(line)
#                         if i >= 500:
#                             break
#             except:
#                 print("wildjailbreak.jsonl not found")
#                 return
#
#             unsafe_pair = []
#             safe_pair = []
#
#             template = self.one_shot_template
#
#             for d in train_data:
#
#                 prompt = d.get("prompt", "")
#                 harmful = d.get("harmful_answer", "")
#                 harmless = d.get("harmless_answer", "")
#                 dtype = d.get("prompt_type", "")
#
#                 if "harmful" in dtype.lower():
#                     if harmful and harmless:
#                         unsafe_pair.append(
#                             template.format(
#                                 user_tag=self.user_tag,
#                                 assistant_tag=self.assistant_tag,
#                                 instruction=prompt,
#                                 response=harmful
#                             )
#                         )
#
#                         safe_pair.append(
#                             template.format(
#                                 user_tag=self.user_tag,
#                                 assistant_tag=self.assistant_tag,
#                                 instruction=prompt,
#                                 response=harmless
#                             )
#                         )
#
#                 if len(unsafe_pair) >= 500:
#                     break
#
#             self.unsafe_prompt_pair_unsafe_answer = unsafe_pair
#             self.unsafe_prompt_pair_safe_answer = safe_pair
#
#             print(f"WildJailbreak loaded: {len(unsafe_pair)}")
#
#         # =========================================================
#         # RUN PIPELINE
#         # =========================================================
#         self.one_shot_template = set_tags_templates()
#
#         sample_from_ultrachat()
#         sample_from_do_not_answer_with_kmeans()
#         sample_from_wildjailbreak()
#
#         self.tokenizer.padding_side = "right"
#
#     # =========================================================
#     def __len__(self):
#         return min(
#             len(self.data_safe_samples),
#             len(self.data_unsafe_samples),
#             len(self.retain_set),
#             len(self.unsafe_prompt_pair_unsafe_answer)
#         )
#
#     # =========================================================
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         safe = self.data_safe_samples[i]
#         unsafe = self.data_unsafe_samples[i]
#         label = self.data_unsafe_labels[i]
#         retain = self.retain_set[i]
#         unsafe_pair = self.unsafe_prompt_pair_unsafe_answer[i]
#         safe_pair = self.unsafe_prompt_pair_safe_answer[i]
#
#         def tok(x, max_len):
#             return self.tokenizer(
#                 x,
#                 max_length=max_len,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             )
#
#         half = self.max_length // 2
#
#         # <SEPARATOR> 기준으로 request/response 분리
#         safe_req, safe_res = safe.split("<SEPARATOR>")
#         unsafe_req, unsafe_res = unsafe.split("<SEPARATOR>")
#         ur_req, ur_res = unsafe_pair.split("<SEPARATOR>")  # unsafe request + unsafe response
#         sr_req, sr_res = safe_pair.split("<SEPARATOR>")  # unsafe request + safe response
#
#         # 토크나이징
#         safe_in = tok(safe_req, half)
#         safe_out = tok(safe_res, half)
#         unsafe_in = tok(unsafe_req, half)
#         unsafe_out = tok(unsafe_res, half)
#         retain_tok = tok(retain, self.max_length)
#         ur_in = tok(ur_req, half)
#         ur_out = tok(ur_res, half)
#         sr_out = tok(sr_res, half)
#
#         return {
#             # safe sample (safe request + safe response)
#             "ids_safe_sample": torch.cat([safe_in["input_ids"], safe_out["input_ids"]], dim=1),
#             "mask_safe_sample": torch.cat([safe_in["attention_mask"], safe_out["attention_mask"]], dim=1),
#             "mask_safe_sample_request": safe_in["attention_mask"],
#             "mask_safe_sample_response": safe_out["attention_mask"],
#
#             # unsafe sample (unsafe request + unsafe response)
#             "ids_unsafe_sample": torch.cat([unsafe_in["input_ids"], unsafe_out["input_ids"]], dim=1),
#             "mask_unsafe_sample": torch.cat([unsafe_in["attention_mask"], unsafe_out["attention_mask"]], dim=1),
#             "mask_unsafe_sample_request": unsafe_in["attention_mask"],
#             "mask_unsafe_sample_response": unsafe_out["attention_mask"],
#
#             # retain set (일반 지식 보존용)
#             "ids_retain": retain_tok["input_ids"],
#             "mask_retain": retain_tok["attention_mask"],
#
#             # unsafe request + unsafe response 쌍
#             "ids_unsafe_request_unsafe_response": torch.cat([ur_in["input_ids"], ur_out["input_ids"]], dim=1),
#             "mask_unsafe_request_unsafe_response": torch.cat([ur_in["attention_mask"], ur_out["attention_mask"]],
#                                                              dim=1),
#             "mask_unsafe_response_for_unsafe_request": ur_out["attention_mask"],
#
#             # unsafe request + safe response 쌍
#             "ids_unsafe_request_safe_response": torch.cat([ur_in["input_ids"], sr_out["input_ids"]], dim=1),
#             "mask_unsafe_request_safe_response": torch.cat([ur_in["attention_mask"], sr_out["attention_mask"]], dim=1),
#             "mask_safe_response_for_unsafe_request": sr_out["attention_mask"],
#
#             # unsafe request 단독
#             "ids_unsafe_request": ur_in["input_ids"],
#             "mask_unsafe_request": ur_in["attention_mask"],
#
#             # CRUSH 클러스터 라벨
#             "labels": torch.tensor(label, dtype=torch.long).unsqueeze(0),
#         }

import random
import os
from typing import Dict

import jsonlines
import torch
import transformers
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests
import time

import random
import os
from typing import Dict

import jsonlines
import torch
import transformers
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import requests
import time

random.seed(0)


class RepBendingDataset(Dataset):

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 num_examples,
                 mode,
                 max_length,
                 model_name_or_path,
                 dataset_path=None,
                 split=None,
                 is_online=False,
                 ):
        super().__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.num_examples = num_examples

        self.data_safe_samples = []       # harmless prompt + harmless answer
        self.data_unsafe_samples = []     # harmful prompt + harmful answer
        self.data_unsafe_labels = []      # 클러스터 라벨 (0~4)
        self.retain_set = []              # KL Divergence용 일반 텍스트
        self.unsafe_prompt_pair_unsafe_answer = []  # harmful prompt + harmful answer (wildjailbreak)
        self.unsafe_prompt_pair_safe_answer = []    # harmful prompt + harmless answer (wildjailbreak)

        # =========================================================
        # TEMPLATE
        # =========================================================
        def set_tags_templates():
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

            user_tag, assistant_tag = None, None

            if "qwen" in self.model_name_or_path:
                print("USING QWEN TEMPLATE")
                user_tag = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
            elif "llama" in self.model_name_or_path:
                print("USING LLAMA TEMPLATE")
                user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
                assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                raise NotImplementedError(f"Model {self.model_name_or_path} not supported")

            self.user_tag = user_tag
            self.assistant_tag = assistant_tag
            return one_shot_template

        # =========================================================
        # API 요청 헬퍼 (타임아웃 재시도)
        # =========================================================
        def fetch_rows(url, params, max_retries=3):
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=120)
                    return response.json().get("rows", [])
                except requests.exceptions.ReadTimeout:
                    print(f"  타임아웃 발생, {attempt + 1}번째 재시도...")
                    time.sleep(5)
            return []

        # =========================================================
        # ULTRACHAT (Safe: harmless prompt + harmless answer, 500개)
        # =========================================================
        def sample_from_ultrachat():
            print("Loading UltraChat via HuggingFace API (500개)...")

            url = "https://datasets-server.huggingface.co/rows"
            safe_samples = []
            retain = []
            offset = 0

            while len(safe_samples) < 500:
                params = {
                    "dataset": "HuggingFaceH4/ultrachat_200k",
                    "config": "default",
                    "split": "train_sft",
                    "offset": offset,
                    "length": 100
                }

                rows = fetch_rows(url, params)
                if not rows:
                    break

                for row in rows:
                    messages = row.get("row", {}).get("messages", [])
                    if len(messages) < 2:
                        continue

                    instruction = messages[0].get("content", "")
                    response_text = messages[1].get("content", "")

                    if not instruction or not response_text:
                        continue

                    safe_samples.append(
                        self.one_shot_template.format(
                            user_tag=self.user_tag,
                            assistant_tag=self.assistant_tag,
                            instruction=instruction,
                            response=response_text
                        )
                    )
                    retain.append(instruction)

                    if len(safe_samples) >= 500:
                        break

                offset += 100
                print(f"  UltraChat {len(safe_samples)}/500 로드 중...")

            # safe_samples에 추가 (wildjailbreak harmless랑 합쳐질 것)
            self.data_safe_samples.extend(safe_samples)
            self.retain_set.extend(retain)
            print(f"UltraChat loaded: {len(safe_samples)}")

        # =========================================================
        # WILDJAILBREAK
        # Safe: harmless prompt + harmless answer → data_safe_samples (250개)
        # Unsafe: harmful prompt + harmful answer → unsafe_prompt_pair_unsafe_answer (250개)
        #         harmful prompt + harmless answer → unsafe_prompt_pair_safe_answer (250개)
        # =========================================================
        def sample_from_wildjailbreak():
            print("Loading WildJailbreak...")

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            jsonl_path = os.path.join(BASE_DIR, "wildjailbreak.jsonl")

            if not os.path.exists(jsonl_path):
                print("wildjailbreak.jsonl not found, skipping...")
                return

            train_data = []
            try:
                with jsonlines.open(jsonl_path) as f:
                    for i, line in enumerate(f):
                        train_data.append(line)
                        if i >= 1000:  # 여유있게 읽고 필터링
                            break
            except Exception as e:
                print(f"WildJailbreak 읽기 실패: {e}")
                return

            wj_safe_samples = []      # harmless prompt + harmless answer
            wj_unsafe_pair = []       # harmful prompt + harmful answer
            wj_safe_pair = []         # harmful prompt + harmless answer

            template = self.one_shot_template

            for d in train_data:
                prompt = d.get("prompt", "")
                harmful = d.get("harmful_answer", "")
                harmless = d.get("harmless_answer", "")
                dtype = d.get("prompt_type", "")

                if not prompt:
                    continue

                if "harmful" in dtype.lower():
                    # harmful prompt → unsafe pair (harmful answer / harmless answer)
                    if harmful and harmless and len(wj_unsafe_pair) < 250:
                        wj_unsafe_pair.append(
                            template.format(
                                user_tag=self.user_tag,
                                assistant_tag=self.assistant_tag,
                                instruction=prompt,
                                response=harmful
                            )
                        )
                        wj_safe_pair.append(
                            template.format(
                                user_tag=self.user_tag,
                                assistant_tag=self.assistant_tag,
                                instruction=prompt,
                                response=harmless
                            )
                        )

                elif "harmless" in dtype.lower():
                    # harmless prompt → safe_samples에 추가
                    if harmless and len(wj_safe_samples) < 250:
                        wj_safe_samples.append(
                            template.format(
                                user_tag=self.user_tag,
                                assistant_tag=self.assistant_tag,
                                instruction=prompt,
                                response=harmless
                            )
                        )
                        self.retain_set.append(prompt)

                if len(wj_unsafe_pair) >= 250 and len(wj_safe_samples) >= 250:
                    break

            # safe_samples에 합치기 (UltraChat 500 + WildJailbreak harmless 250 = 750)
            self.data_safe_samples.extend(wj_safe_samples)

            # unsafe pair 저장
            self.unsafe_prompt_pair_unsafe_answer = wj_unsafe_pair
            self.unsafe_prompt_pair_safe_answer = wj_safe_pair

            print(f"WildJailbreak safe loaded: {len(wj_safe_samples)}")
            print(f"WildJailbreak unsafe pair loaded: {len(wj_unsafe_pair)}")

        # =========================================================
        # DO NOT ANSWER
        # Unsafe: harmful prompt + harmful(?) answer → data_unsafe_samples (500개)
        # Seeded K-Means로 카테고리 라벨링
        # =========================================================
        def sample_from_do_not_answer_with_kmeans():
            print("Loading Do-Not-Answer via HuggingFace API...")

            # 카테고리 매핑 (라벨 0~4)
            target_categories = {
                "Malicious Uses": 0,
                "Human-Chatbot Interaction Harms": 1,
                "Information Hazards": 2,
                "Misinformation Harms": 3,
                "Discrimination, Exclusion, Toxicity, Hateful, Offensive": 4
            }

            url = "https://datasets-server.huggingface.co/rows"
            raw_unsafe = []
            raw_inst = []

            # 카테고리별 딱 1개씩만 seed로 사용
            seed_prompts = [""] * 5
            seed_collected = set()  # 이미 seed 수집된 카테고리 인덱스

            offset = 0

            while len(raw_unsafe) < 500:
                params = {
                    "dataset": "LibrAI/do-not-answer",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": 100
                }

                rows = fetch_rows(url, params)
                if not rows:
                    break

                for row in rows:
                    r = row.get("row", {})
                    instruction = r.get("question", "")
                    category = r.get("risk_area", "Unknown")

                    if not instruction or category not in target_categories:
                        continue

                    cat_idx = target_categories[category]

                    # harmful answer 추출
                    harmful_answer = ""
                    for model_name in ["GPT4", "ChatGPT", "Claude", "ChatGLM2",
                                       "llama2-7b-chat", "vicuna-7b"]:
                        resp = r.get(f"{model_name}_response", "")
                        is_harmful = r.get(f"{model_name}_harmful", 0)
                        if resp and is_harmful == 1 and not harmful_answer:
                            harmful_answer = resp
                            break

                    if not harmful_answer:
                        continue

                    # 카테고리별 첫 번째 1개만 seed로 수집
                    if cat_idx not in seed_collected:
                        seed_prompts[cat_idx] = instruction
                        seed_collected.add(cat_idx)
                        print(f"  🌱 Seed 수집 [{cat_idx}] {category[:30]}: {instruction[:50]}...")
                        # seed 데이터는 클러스터링 대상에서 제외
                        continue

                    # 나머지는 클러스터링 대상 (카테고리 모름, KMeans가 판단)
                    formatted = self.one_shot_template.format(
                        user_tag=self.user_tag,
                        assistant_tag=self.assistant_tag,
                        instruction=instruction,
                        response=harmful_answer
                    )

                    raw_unsafe.append(formatted)
                    raw_inst.append(instruction)

                    if len(raw_unsafe) >= 500:
                        break

                offset += 100
                print(f"  Do-Not-Answer {len(raw_unsafe)}/500 로드 중... (Seed: {len(seed_collected)}/5)")

            # seed가 덜 모인 경우 경고
            if len(seed_collected) < 5:
                print(f"⚠️  Seed가 {len(seed_collected)}개만 수집됨. 빈 seed는 랜덤 초기화로 대체됩니다.")

            print("\n🌱 수집된 Seed 프롬프트:")
            cat_names = list(target_categories.keys())
            for i, prompt in enumerate(seed_prompts):
                print(f"  [{i}] {cat_names[i][:30]}: {prompt[:60]}...")

            # =========================================================
            # Seeded K-Means:
            # - seed: 카테고리별 딱 1개 프롬프트의 TF-IDF 벡터
            # - 나머지 데이터: 어느 카테고리인지 모름 → KMeans가 판단
            # =========================================================
            print("\nRunning Seeded K-Means clustering...")
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(raw_inst)  # 클러스터링 대상 (seed 제외)

            # seed 벡터 변환
            seed_vectors = vectorizer.transform(seed_prompts).toarray()

            # 빈 seed가 있으면 X의 랜덤 샘플로 대체
            for i in range(5):
                if seed_prompts[i] == "":
                    rand_idx = np.random.randint(0, X.shape[0])
                    seed_vectors[i] = X[rand_idx].toarray()[0]

            pseudo_labels = KMeans(
                n_clusters=5,
                init=seed_vectors,  # 카테고리별 1개 프롬프트로 초기화
                n_init=1,
                random_state=42
            ).fit_predict(X)

            print(f"클러스터링 완료! 라벨 분포: {np.bincount(pseudo_labels)}")

            self.data_unsafe_samples = raw_unsafe
            self.data_unsafe_labels = pseudo_labels.tolist()

            print(f"Do-Not-Answer loaded: {len(raw_unsafe)}")
            print(f"라벨 분포: {np.bincount(pseudo_labels)}")

        # =========================================================
        # RUN PIPELINE
        # =========================================================
        self.one_shot_template = set_tags_templates()

        # 1. Safe 데이터: UltraChat(500) + WildJailbreak harmless(250)
        sample_from_ultrachat()       # → data_safe_samples에 500개 추가
        sample_from_wildjailbreak()   # → data_safe_samples에 250개 추가, unsafe_pair에 250개
        # 2. Unsafe 데이터: Do-Not-Answer(500) + Seeded K-Means 라벨링
        sample_from_do_not_answer_with_kmeans()  # → data_unsafe_samples에 500개

        self.tokenizer.padding_side = "right"

        print(f"\n✅ 데이터셋 구성 완료:")
        print(f"  Safe samples (UltraChat+WJ harmless): {len(self.data_safe_samples)}")
        print(f"  Unsafe samples (Do-Not-Answer):       {len(self.data_unsafe_samples)}")
        print(f"  Unsafe pair (WJ harmful):             {len(self.unsafe_prompt_pair_unsafe_answer)}")
        print(f"  Retain set:                           {len(self.retain_set)}")

    # =========================================================
    def __len__(self):
        return min(
            len(self.data_safe_samples),
            len(self.data_unsafe_samples),
            len(self.retain_set),
            len(self.unsafe_prompt_pair_unsafe_answer)
        )

    # =========================================================
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        safe = self.data_safe_samples[i]
        unsafe = self.data_unsafe_samples[i]
        label = self.data_unsafe_labels[i]
        retain = self.retain_set[i]
        unsafe_pair = self.unsafe_prompt_pair_unsafe_answer[i]
        safe_pair = self.unsafe_prompt_pair_safe_answer[i]

        def tok(x, max_len):
            return self.tokenizer(
                x,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        half = self.max_length // 2

        # <SEPARATOR> 기준으로 request/response 분리
        safe_req, safe_res = safe.split("<SEPARATOR>")
        unsafe_req, unsafe_res = unsafe.split("<SEPARATOR>")
        ur_req, ur_res = unsafe_pair.split("<SEPARATOR>")   # harmful prompt + harmful answer
        sr_req, sr_res = safe_pair.split("<SEPARATOR>")     # harmful prompt + harmless answer

        # 토크나이징
        safe_in   = tok(safe_req, half)
        safe_out  = tok(safe_res, half)
        unsafe_in = tok(unsafe_req, half)
        unsafe_out = tok(unsafe_res, half)
        retain_tok = tok(retain, self.max_length)
        ur_in  = tok(ur_req, half)
        ur_out = tok(ur_res, half)
        sr_out = tok(sr_res, half)

        return {
            # safe sample: harmless prompt + harmless answer
            "ids_safe_sample":              torch.cat([safe_in["input_ids"],  safe_out["input_ids"]],  dim=1),
            "mask_safe_sample":             torch.cat([safe_in["attention_mask"], safe_out["attention_mask"]], dim=1),
            "mask_safe_sample_request":     safe_in["attention_mask"],
            "mask_safe_sample_response":    safe_out["attention_mask"],

            # unsafe sample: harmful prompt + harmful answer (do-not-answer)
            "ids_unsafe_sample":            torch.cat([unsafe_in["input_ids"],  unsafe_out["input_ids"]],  dim=1),
            "mask_unsafe_sample":           torch.cat([unsafe_in["attention_mask"], unsafe_out["attention_mask"]], dim=1),
            "mask_unsafe_sample_request":   unsafe_in["attention_mask"],
            "mask_unsafe_sample_response":  unsafe_out["attention_mask"],

            # retain set
            "ids_retain":                   retain_tok["input_ids"],
            "mask_retain":                  retain_tok["attention_mask"],

            # harmful prompt + harmful answer (wildjailbreak)
            "ids_unsafe_request_unsafe_response":       torch.cat([ur_in["input_ids"],  ur_out["input_ids"]],  dim=1),
            "mask_unsafe_request_unsafe_response":      torch.cat([ur_in["attention_mask"], ur_out["attention_mask"]], dim=1),
            "mask_unsafe_response_for_unsafe_request":  ur_out["attention_mask"],

            # harmful prompt + harmless answer (wildjailbreak)
            "ids_unsafe_request_safe_response":         torch.cat([ur_in["input_ids"],  sr_out["input_ids"]],  dim=1),
            "mask_unsafe_request_safe_response":        torch.cat([ur_in["attention_mask"], sr_out["attention_mask"]], dim=1),
            "mask_safe_response_for_unsafe_request":    sr_out["attention_mask"],

            # harmful prompt 단독
            "ids_unsafe_request":           ur_in["input_ids"],
            "mask_unsafe_request":          ur_in["attention_mask"],

            # CRUSH 클러스터 라벨
            "labels":                       torch.tensor(label, dtype=torch.long).unsqueeze(0),
        }