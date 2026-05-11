# CRUSH: Cluster-based Representation Bending for LLM Safety

> **C**luster-based **R**epresentation **U**pdate for **S**afety **H**andling  
> LLM의 latent space에서 유해 프롬프트를 카테고리별로 분리하여, 무엇이 위험한지 모델 자신이 표상 수준에서 구분하게 만드는 unsupervised contrastive learning 프로젝트.

---

## 1. 프로젝트 개요

### 1.1 문제의식
기존 LLM safety 학습은 "유해/무해" 이분법으로 모델을 정렬하지만, 유해 프롬프트도 종류가 다양하다 (범죄 모의, 차별 발언, 정서적 가스라이팅 등). 모델이 **각 유해 카테고리를 latent space 상 별개의 영역으로 분리**하면:
- 어떤 종류의 위해인지 표상 수준에서 식별 가능
- 카테고리별로 다른 안전 처리 적용 가능
- 사람이 정답 라벨을 매기지 않은 새로운 프롬프트에도 일반화 가능

### 1.2 핵심 아이디어
1. **사전 클러스터링**: 학습 시작 전 base 모델로 유해 프롬프트의 hidden state를 추출 → K-Means로 5개 클러스터 형성 → 각 클러스터의 평균을 centroid로 고정
2. **Contrastive 학습**: LoRA로 fine-tuning하며, 같은 클러스터 샘플은 자기 centroid로 끌어당기고(pull), 다른 클러스터 centroid에서는 밀어내기(push)
3. **검증**: 학습 후 latent에서 다시 K-Means를 돌려 정답 라벨(risk_area)과의 NMI/ARI를 측정 → "라벨 없이 학습했는데 의미 카테고리가 자연 발생했는가" 검증

### 1.3 기술적 핵심 결정사항
| # | 결정 | 근거 |
|---|------|------|
| ❶ | DNA만 pull/push 학습, WJ는 KL/Benign loss에만 참여 | DNA는 카테고리 정보 있음, WJ는 jailbreak 형식이 우선 |
| ❷ | 학습 시작 전 1회 클러스터링 (epoch마다 재클러스터링 X) | 캡스톤 일정 + 안정성 |
| ❸ | Centroid 사전 계산 후 학습 내내 고정 | EMA collapse 방지 (이전 버전 실패 원인) |
| ❹ | 클러스터링 공간 = target_layers의 중간 레이어 hidden | LLM의 중후반 hidden이 의미 정보 인코딩 |
| ❺ | Hidden 추출 위치 = request 마지막 토큰 | 사용자 프롬프트의 의미가 가장 잘 응축된 지점 |

---

## 2. 사용 데이터셋

총 3개의 공개 데이터셋을 조합한다. 각 데이터셋은 서로 다른 역할을 수행한다.

### 2.1 UltraChat 200k (HuggingFaceH4/ultrachat_200k)

| 항목 | 내용 |
|------|------|
| **링크** | https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k |
| **라이선스** | MIT |
| **전체 규모** | train_sft 207,865개 / test_sft 23,110개 등 총 약 50만 dialog |
| **이 프로젝트에서 사용** | **train_sft에서 939개** (다음 두 데이터셋과 개수 통일) |
| **역할** | Safe sample (무해 프롬프트 + 무해 응답) |
| **로드 방식** | HuggingFace Datasets Server API (`datasets-server.huggingface.co/rows`) |

**구성**: ChatGPT가 생성한 1.4M dialog를 필터링한 SFT용 데이터셋. 이 프로젝트에서는 첫 메시지를 prompt로, 두 번째 메시지를 response로 사용해 "정상 대화" 페어를 만든다.

**사용 인용**:
> Ding et al., 2023. "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations." arXiv:2305.14233

### 2.2 Do-Not-Answer (LibrAI/do-not-answer)

| 항목 | 내용 |
|------|------|
| **링크** | https://huggingface.co/datasets/LibrAI/do-not-answer |
| **GitHub** | https://github.com/Libr-AI/do-not-answer |
| **라이선스** | Apache 2.0 |
| **전체 규모** | **939개** instruction (5개 risk_area × 12 harm types × 61 specific harms) |
| **이 프로젝트에서 사용** | 전체 939개 |
| **역할** | Unsafe sample (DNA) + **클러스터링/평가의 정답 라벨 공급원** |
| **로드 방식** | HuggingFace Datasets Server API |

**핵심 필드**:
- `question` : 위해 프롬프트 (예: "How can I create fake credentials...")
- `risk_area` : 5개 카테고리 정답 라벨
  - `Malicious Uses` → label 0 (범죄 모의·악용)
  - `Human-Chatbot Interaction Harms` → label 1 (정서적 의존·가스라이팅)
  - `Information Hazards` → label 2 (위험 정보 유출)
  - `Misinformation Harms` → label 3 (허위 사실·음모론)
  - `Discrimination, Exclusion, Toxicity, Hateful, Offensive` → label 4 (차별·혐오)
- `{model_name}_response` : 6개 LLM (GPT-4, ChatGPT, Claude, ChatGLM2, LLaMA-2, Vicuna)의 응답 → 가장 긴 응답을 학습용 response로 채택

**이 프로젝트에서의 활용**:
1. 학습용 unsafe 페어 939개 → DNA prompt + (LLM 응답 중 최장)
2. 클러스터링 대상 (DNA prompt만 K-Means에 들어감)
3. `risk_area`는 학습에는 사용 안 함, **검증 단계에서만** Before/After NMI·ARI·Hungarian Accuracy 계산용 정답으로 사용

**사용 인용**:
> Wang et al., 2023. "Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs." arXiv:2308.13387

### 2.3 WildJailbreak (allenai/wildjailbreak)

| 항목 | 내용 |
|------|------|
| **링크** | https://huggingface.co/datasets/allenai/wildjailbreak |
| **라이선스** | ODC-BY (게이트 접근, allenai 가입 필요) |
| **전체 규모** | 262K prompt-response 페어 (vanilla 100K + adversarial 162K) |
| **이 프로젝트에서 사용** | **939개씩** (harmful prompt + 무해/유해 두 응답 페어) |
| **역할** | Jailbreak 방어 학습용 contrastive 페어 |
| **로드 방식** | 사전 다운로드 후 `wildjailbreak.jsonl` 로 변환 (게이트 접근 때문) |

**핵심 필드** (변환된 jsonl 기준):
- `prompt` : 우회/롤플레이/시나리오 등으로 jailbreak를 시도하는 프롬프트
- `prompt_type` : "vanilla_harmful" / "vanilla_benign" / "adversarial_harmful" / "adversarial_benign"
- `harmful_answer` : 모델이 jailbreak에 굴복했을 경우의 위험한 응답
- `harmless_answer` : 동일 프롬프트에 대한 안전한 거부 응답

**이 프로젝트에서의 활용**:
- `harmful` 타입에서 939개 추출 → 같은 jailbreak prompt에 대한 (harmful_answer, harmless_answer) 페어 형성
- `harmless` 타입에서 939개 추출 → safe sample 풀에 합류 (UltraChat 939 + WJ harmless 939 = 1878 safe samples)
- **WJ에는 risk_area 같은 카테고리 라벨이 없음** → pull/push 학습에서는 제외하고 KL/Benign loss에만 참여

**사용 인용**:
> Jiang et al., 2024. "WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models." NeurIPS 2024.

### 2.4 데이터셋 활용 요약

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         학습 데이터 구성                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Safe Samples (총 1878개) ← KL/Benign loss 학습용                        │
│    ├─ UltraChat 939개 (정상 대화)                                        │
│    └─ WildJailbreak harmless 939개 (위험해 보이는 무해 프롬프트)         │
│                                                                          │
│  Unsafe Samples (총 939개) ← Pull/Push 학습 + 검증                       │
│    └─ Do-Not-Answer 939개 (5개 risk_area 카테고리)                       │
│                                                                          │
│  Unsafe Pair (총 939개) ← Jailbreak 방어용                               │
│    └─ WildJailbreak harmful 939개                                        │
│        - Harmful prompt + Harmful answer                                 │
│        - Harmful prompt + Harmless answer  ← contrastive pair            │
│                                                                          │
│  Retain Set (총 1878개) ← KL Divergence loss 학습용                      │
│    └─ Safe 프롬프트와 동일 (일반 능력 보존)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 로직: 어떻게 작동하는가

### 3.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      [Phase 1: 데이터 로드]                              │
│  RepBendingDataset 생성                                                  │
│   ├─ UltraChat 939 로드 (HF API)                                         │
│   ├─ WildJailbreak 939+939 로드 (로컬 jsonl)                             │
│   └─ Do-Not-Answer 939 로드 (HF API) + risk_area 정답 라벨 보존          │
│  ※ 이 시점에 pseudo-label은 아직 없음                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  [Phase 2: 사전 클러스터링] ← 학습 전 1회                │
│  clustering.cluster_unsafe_prompts()                                     │
│   1. Base 모델로 DNA 프롬프트 939개 → cluster_layer hidden 추출          │
│      (cluster_layer = target_layers의 중간, 예: layer 17)                │
│   2. L2 정규화 후 K-Means(k=5, n_init=10) 실행                           │
│   3. 각 클러스터의 hidden 평균 → centroid 5개 (L2 정규화 후 고정)        │
│   4. 진단 출력:                                                          │
│      - 클러스터별 샘플 분포                                              │
│      - risk_area와의 NMI/ARI (사전 측정)                                 │
│      - 클러스터별 대표 프롬프트 5개 (사람이 의미 검증)                   │
│   5. dataset.set_pseudo_labels() 주입                                    │
│   6. crush_centroids_initial.pt 저장 (검증 단계에서 사용)                │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    [Phase 3: LoRA 학습]                                  │
│  CustomTrainer.train()                                                   │
│   매 스텝마다:                                                            │
│     1. Forward 통과 (safe / unsafe / retain 배치)                        │
│     2. Loss 계산:                                                        │
│        - Benign  = α·max(0, |orig-new|_b - min_dist_to_centroid + m_b)   │
│        - Pull    = β·max(0, |new-my_centroid| - |new-orig|_h + m_pull)   │
│        - Push    = γ·max(0, |new-my_centroid| - |new-other_centroid|+m_p)│
│        - KL      = ε·KL(retain | original_retain)                        │
│        - Total   = α·Benign + β·Pull + γ·Push + ε·KL                     │
│     3. Backprop으로 LoRA 가중치만 업데이트                               │
│     4. Centroid는 변화 없음 (고정)                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    [Phase 4: 검증/평가]                                  │
│  evaluate.py                                                             │
│   1. 학습 후 모델로 평가 데이터의 hidden 재추출 (Before/After 비교)      │
│   2. risk_area 정답 라벨 기준:                                           │
│      - Silhouette / Davies-Bouldin (분리도)                              │
│   3. 학습된 latent에서 새 K-Means 실행 → 정답 라벨과 비교:               │
│      - NMI / ARI / Hungarian Accuracy                                    │
│   4. t-SNE 시각화 (Before vs After, true_label로 색칠)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Loss 함수 상세 (`crush_loss.calc_crush_loss`)

모든 hidden vector를 L2 정규화 (단위 구 위에서 작업).

#### Benign Loss
```
loss_benign = mean( ReLU(|h_safe_orig - h_safe_new| - min_dist(h_safe_new, all_centroids) + m_b) )
```
**의미**: Safe 벡터가 원본 위치보다 가장 가까운 유해 클러스터에 더 가까워지면 패널티.  
**역할**: 안전한 프롬프트는 학습 후에도 유해 영역으로 끌려가지 말아야 함.

#### Pull Loss
```
loss_pull = mean( ReLU(|h_harm_new - my_centroid| - |h_harm_new - h_harm_orig| + m_pull) )
```
**의미**: 자기 클러스터 centroid까지의 거리가 원본 위치까지의 거리보다 m_pull 이상 가까워야 함.  
**역할**: 유해 프롬프트를 같은 카테고리끼리 모음.

#### Push Loss
```
loss_push = mean( ReLU(|h_harm_new - my_centroid| - min_dist(h_harm_new, other_centroids) + m_push) )
```
**의미**: 자기 centroid가 가장 가까운 다른 centroid보다 m_push 이상 가까워야 함.  
**역할**: 다른 카테고리끼리 latent에서 분리되도록.

#### KL Divergence Loss
```
loss_kl = KL( softmax(lora_logits / T) || softmax(original_logits / T) )
```
**의미**: Retain set에 대해서는 LoRA 적용 모델의 logits가 원본과 너무 멀어지지 않도록.  
**역할**: 모델의 일반 능력 보존 (catastrophic forgetting 방지).

#### Margin 값
`m_b = m_pull = m_push = 0.5` (이전 버전 0.2에서 변경).  
단위 구 위 5개 점의 이론적 평균 거리(~1.58) 대비 너무 좁으면 push가 즉시 0으로 만족되어 분리 압력이 사라지는 문제 해결.
---

## 4. 코드 구조

### 4.1 파일 트리

```
crush/
├── README.md                    ← 이 문서
├── args.py                      ← 학습 인자 정의 (HyperparamArgs, LoraArgs, ...)
├── dataset.py                   ← RepBendingDataset (3개 데이터셋 로드/통합)
├── clustering.py                ← 학습 전 클러스터링 모듈 (신규)
├── crush_loss.py                ← FixedCentroidHolder + calc_crush_loss
├── trainer.py                   ← CustomTrainer (HF Trainer 확장)
├── classifier.py                ← Online jailbreak 분류기 (HarmbenchClassifier)
├── utils.py                     ← 온라인 샘플 생성 등 유틸
├── train.py                     ← 학습 진입점
├── evaluate.py                  ← 검증/시각화 진입점
└── wildjailbreak.jsonl          ← WJ 데이터 로컬 사본 (게이트 접근 우회)
```

### 4.2 파일별 책임

#### `args.py`
- `HyperparamArguments`: target_layers, loss_alpha/beta/gamma/epsilon, loss_mode 등
- `LoraArguments`: lora_r, lora_alpha, target_modules 등
- `ModelArguments`, `TrainingArguments`

#### `dataset.py` — `RepBendingDataset`
- **3개 데이터셋 로드**: `sample_from_ultrachat()`, `sample_from_wildjailbreak()`, `sample_from_do_not_answer()`
- **보존 필드**:
  - `data_safe_samples` : Safe 학습용 (UltraChat + WJ harmless)
  - `data_unsafe_samples` : Unsafe 학습용 (DNA, full template)
  - `data_unsafe_request_prompts` : DNA의 request만 (clustering.py가 사용)
  - `data_unsafe_true_labels` : DNA의 risk_area 정답 (검증용으로만)
  - `data_unsafe_labels` : pseudo-label (외부 주입 대기)
  - `unsafe_prompt_pair_unsafe_answer` / `unsafe_prompt_pair_safe_answer` : WJ contrastive 페어
  - `retain_set` : KL Divergence용 일반 텍스트
- **주입 메서드**: `set_pseudo_labels(labels)` — clustering.py 결과를 학습용 라벨로 설정
- **`__getitem__`**: pseudo-label 기준 카테고리 균등 샘플링

#### `clustering.py` (신규) — 학습 전 사전 클러스터링
- `extract_request_hidden()` : base 모델로 request 마지막 토큰 hidden 추출
- `run_kmeans_clustering()` : K-Means (n_init=10으로 robust)
- `compute_centroids_from_hidden()` : 클러스터 평균 → L2 정규화 → 고정 centroid
- `diagnose_clustering()` : 분포·NMI·ARI·대표 프롬프트 출력
- `cluster_unsafe_prompts()` : 전체 파이프라인 + hidden 캐싱

#### `crush_loss.py`
- `FixedCentroidHolder` : 사전 계산된 centroid 보관/이동만 담당, 업데이트 없음
- `calc_crush_loss()` : Benign / Pull / Push loss 계산 (margin 0.5)

#### `trainer.py` — `CustomTrainer` (SFTTrainer 확장)
- `_compute_loss()` : 배치를 safe/unsafe/retain으로 분리, DNA 배치 크기 명시
- `_calc_loss()` : 
  - LoRA on/off forward → hidden 추출
  - DNA 부분(앞 절반)만 pull/push에 사용 (WJ는 제외)
  - margin 0.5의 calc_crush_loss 호출
- `DebugLogger` : 매 스텝의 loss·vec_change·centroid 거리 등 로그

#### `train.py` — 학습 진입점
1. Args 파싱
2. Tokenizer / Dataset 생성
3. Base 모델 로드 (4bit quantization)
4. **★ clustering.py로 pseudo-label + centroid 사전 계산**
5. LoRA 적용
6. CustomTrainer 생성 (centroid_holder 주입)
7. 학습 시작
8. 학습 후 저장: `crush_centroids.pt`, `crush_centroids_initial.pt`

#### `evaluate.py` — 검증 진입점
1. LoRA 어댑터 + centroid + cluster_layer 메타 로드
2. 평가 데이터 50개씩 균등 추출 (risk_area 기준)
3. Before/After × cluster_layer hidden 추출
4. `evaluate_separation()` : true-label 기준 Silhouette/D-B
5. `evaluate_clustering_recovery()` : 새 K-Means → NMI/ARI/Hungarian Accuracy
6. t-SNE 시각화 (Before vs After, risk_area 색칠)

---

## 5. 실행 방법

### 5.1 환경 준비

**Python 3.10+, PyTorch 2.x, CUDA 환경 (4bit quantization 사용)**

```bash
pip install transformers peft trl bitsandbytes accelerate
pip install scikit-learn scipy matplotlib numpy
pip install datasets requests jsonlines
```

### 5.2 WildJailbreak 데이터 준비

게이트 접근 데이터셋이므로 사전 다운로드 필요:
1. https://huggingface.co/datasets/allenai/wildjailbreak 에서 접근 신청
2. 다운로드 후 `wildjailbreak.jsonl` 형식으로 변환 (각 줄: `{"prompt": ..., "prompt_type": ..., "harmful_answer": ..., "harmless_answer": ...}`)
3. `dataset.py`와 같은 디렉토리에 배치

### 5.3 학습 실행

```bash
python train.py \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --loss_alpha 0.5 \
  --loss_beta 1.5 \
  --loss_gamma 3.0 \
  --loss_epsilon 0.1 \
  --loss_eta 0.0 \
  --loss_mode response_all \
  --output_dir ./out/test \
  --max_steps 500 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-4 \
  --max_seq_length 512 \
  --num_train_epochs 1 \
  --report_to none \
  --bf16 False \
  --fp16 True
```

**자동 동작**:
- `target_layers`를 안 넘기면 모델 전체 레이어의 중후반(50%~끝)이 자동 선택
- `cluster_layer = target_layers의 중간 인덱스`
- 학습 시작 전에 클러스터링 진단 출력 (NMI, 분포, 대표 프롬프트)
- 학습 후 `./out/test/`에 LoRA 가중치 + 두 centroid 파일 저장

### 5.4 평가 실행

```bash
python evaluate.py
```

**출력**:
- Before vs After 분리도 지표 (Silhouette, D-B)
- Before vs After 클러스터 회복도 (NMI, ARI, Hungarian Accuracy)
- t-SNE 그림 (`crush_tsne_before_after_train.png`)

---

## 6. 핵심 검증 가설과 평가

### 6.1 검증 가설

> **H1**: 학습 후 latent에서 새 K-Means를 돌리면, true_labels(risk_area)와의 NMI/ARI/Hungarian Accuracy가 학습 전보다 개선된다.

H1이 참 → "라벨 없이 학습했는데 의미 카테고리가 자연 발생함" 입증.  
H1이 거짓 → pseudo-label이 true-label과 잘 안 맞아 학습이 의미 분리를 못 만든 것.

### 6.2 평가 지표 해석 기준

| 지표 | 범위 | 좋은 값 | 의미 |
|------|------|---------|------|
| Silhouette | [-1, 1] | > 0.3 양호, > 0.5 우수 | 클러스터 내 응집 vs 클러스터 간 분리 |
| Davies-Bouldin | [0, ∞) | 낮을수록 좋음 | 클러스터 유사도 평균 |
| NMI | [0, 1] | > 0.3 의미 있음 | pseudo와 true의 mutual information |
| ARI | [-1, 1] | > 0.3 의미 있음 | 페어 단위 일치도 |
| Hungarian Acc | [0, 1] | > 0.6 양호 | 최적 permutation 후 분류 정확도 |

### 6.3 결과 분석 관점
- **NMI/ARI 개선** → pseudo-label로 학습했어도 의미 분리가 강화됨 (H1 입증)
- **NMI/ARI 정체/악화** → pseudo-label이 true-label과 부분 일치만 해서 학습이 일부는 의미 분리, 일부는 노이즈 학습  
- **t-SNE 시각적 개선 + NMI 정체**: K-Means(k=5) 평가의 한계. 한 카테고리가 두 클러스터로 쪼개지면 NMI는 떨어지지만 시각적으로는 분리 명확
- 발표 시: NMI 단일 수치만이 아니라 **시각화 + 카테고리별 confusion matrix + Silhouette 변화**를 종합 제시

---

## 7. 단위 테스트 통과 항목

- ✅ K-Means → centroid 계산 → L2 정규화 (norms ≈ 1.0)
- ✅ FixedCentroidHolder가 1000 스텝 backprop 후 변화량 0 (완전 고정 검증)
- ✅ NMI/ARI/Hungarian Accuracy = 1.0 (이상적 mock 데이터)
- ✅ DNA/WJ 슬라이싱 후 loss 계산 정상
- ✅ labels = -1 (pseudo-label 미주입) 시 ValueError로 명시 차단
- ✅ 모든 파일 syntax + cross-module import 일치

---

## 8. 라이선스 및 인용

### 8.1 사용 데이터셋 라이선스
- UltraChat 200k : MIT
- Do-Not-Answer : Apache 2.0
- WildJailbreak : ODC-BY (Open Data Commons)

각 데이터셋의 원본 인용은 §2에 명시.

### 8.2 본 프로젝트 인용
연구 결과를 외부에 공유할 경우 위 3개 원본 데이터셋과 다음 base 방법론들을 함께 인용 권장:
- RepBend (representation bending) 계열 LLM safety 연구
- Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
- Caron et al., "DeepCluster" (ECCV 2018)

---

## 9. 알려진 한계 및 향후 개선

### 9.1 현 버전의 한계
- **K-Means 클러스터링이 base 모델 hidden 공간에서 risk_area를 부분적으로만 잡음** (사전 NMI ~0.26)
  - 학습 후 일부 카테고리(Malicious, Discrim)는 명확히 분리되지만 다른 카테고리(ChatbotHarm, Misinfo)는 분산
- **학습 진행 중 hidden 분포가 변하면서 초기 pseudo-label이 stale해질 수 있음**
- WJ 데이터는 카테고리 분리 학습에서 빠짐 (데이터 절반 미활용)

### 9.2 향후 개선 방향
- **DeepCluster 패턴**: epoch마다 hidden 재추출 → K-Means 재실행 → pseudo-label 갱신
- **Cluster_layer 탐색**: 현재 target_layers 중간 → 더 추상적인 후반 레이어로 변경 가능성
- **WJ에 zero-shot 카테고리 분류기 적용**: GPT-4o-mini 등으로 WJ 프롬프트에 카테고리 부여 후 학습 풀에 추가
- **Supervised baseline 추가 실험**: risk_area 정답을 직접 학습 라벨로 사용 → unsupervised와 정량 비교
- **Margin / loss weight 추가 탐색**

---

## 10. 팀 정보

**팀 0123 — 캡스톤 프로젝트**  
생성형 AI 정신 건강 위해성 안전 제어 아키텍처

| 역할 | 이름 |
|------|------|
| 팀원 | 우민하, 안진영, 임재모 |
| 기간 | 캡스톤 진행 중 |
| 베이스 모델 | Qwen/Qwen2-0.5B-Instruct (4bit quantization) |
| 학습 방식 | LoRA fine-tuning |
