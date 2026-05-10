# CRUSH: Representation Bending for Safe LLM Alignment

본 프로젝트는 LoRA(Low-Rank Adaptation)를 활용하여 대형 언어 모델(LLM)의 잠재 공간(Latent Space)을 재구성하는 **CRUSH (Clustering Representations for Unsafe Speech Halting)** 방법론의 구현체입니다. 유해한 프롬프트(Harmful Prompts)의 표현을 특정 카테고리별로 클러스터링하고, 안전한 프롬프트(Benign Prompts)와의 거리를 벌려 모델이 유해한 요청을 효과적으로 차단하면서도 일반적인 지식은 보존하도록 학습합니다.

---

## 1. CRUSH Loss 수식 및 원리 설명

학습의 핵심은 `crush_loss.py`에 구현된 3가지 마진 기반 손실 함수(Margin-based Loss)와 일반 성능 보존을 위한 KL Divergence 손실입니다.

### Benign Loss ($L_{benign}$)
안전한 요청($h_{sa}$)의 벡터가 유해 클러스터 중심점($c$)들로부터 최소 마진($m_{b}$) 이상 떨어지도록 밀어냅니다. 정상적인 답변 공간을 보호하는 역할을 합니다.
$$L_{benign} = \frac{1}{N} \sum \max(0, m_{b} - \min_{c} ||h_{sa} - c||)$$

### Pull Loss ($L_{pull}$)
유해한 요청($h_{un}$)의 벡터를 자신이 속한 타겟 카테고리의 중심점($c_{target}$)으로 당깁니다. 단, 원래 모델에서의 거리($h_{un}^{orig}$)를 고려하여 무한정 당기지 않고 구조를 유지합니다.
$$L_{pull} = \frac{1}{N} \sum \max(0, ||h_{un} - c_{target}|| - ||h_{un} - h_{un}^{orig}|| + m_{pull})$$

### Push Loss ($L_{push}$)
유해한 요청($h_{un}$)의 벡터가 자신이 속하지 않은 다른 카테고리의 중심점들($c \neq target$)과는 거리를 벌려, 클러스터 간의 분리도를 높입니다.
$$L_{push} = \frac{1}{N} \sum \max(0, ||h_{un} - c_{target}|| - \min_{c \neq target} ||h_{un} - c|| + m_{push})$$

### Total Loss
최종 손실 함수는 위 세 가지 Loss와 기존 모델의 출력 분포를 유지하기 위한 KL Divergence를 합산하여 계산됩니다.
$$Loss_{total} = \alpha L_{benign} + \beta L_{pull} + \gamma L_{push} + \epsilon L_{KL}$$

---

## 2. 하이퍼파라미터 및 주요 지표

### 주요 하이퍼파라미터 (`args.py`)
* **loss_alpha, loss_beta, loss_gamma**: 각각 Benign, Pull, Push Loss에 곱해지는 가중치입니다.
* **loss_epsilon**: 모델의 일반 성능 보존(Retain set)을 위한 KL Divergence Loss의 가중치입니다.
* **target_layers / transform_layers**: Representation을 추출하고 LoRA를 적용할 모델의 레이어 인덱스입니다 (`-1` 입력 시 중간~마지막 레이어로 자동 감지).
* **momentum**: `CentroidManager`에서 클러스터 중심점을 업데이트할 때 사용하는 모멘텀 값(기본 0.9)입니다.

### 평가 지표 (`evaluate.py`, `trainer.py`)
* **Total_Loss**: 위에서 설명한 4가지 Loss의 가중합.
* **Silhouette Score & Davies-Bouldin Index**: 잠재 공간 내 유해 벡터들의 클러스터링 품질을 평가합니다.
* **Vec_Change**: LoRA 적용 전후 유해 벡터의 평균 이동 거리(L2 Norm).
* **Centroid_Dist / Intra_Dist**: 중심점 간의 거리(클러스터 간 분리도) 및 클러스터 내부 벡터 간 거리(응집도).

---

## 3. 실행 방법 (Terminal Commands)

모델 학습 및 평가를 위한 기본 명령어입니다.

### 학습 실행 (Training)

    python train.py \
        --model_name_or_path "Qwen/Qwen2-0.5B-Instruct" \
        --target_layers "-1" \
        --transform_layers "-1" \
        --loss_alpha 1.0 \
        --loss_beta 1.0 \
        --loss_gamma 1.0 \
        --loss_epsilon 0.1 \
        --max_steps 300 \
        --output_dir "./out/crush_model"

### 평가 실행 (Evaluation)
학습된 LoRA 모델과 중심점(`crush_centroids.pt`)을 불러와 성능을 평가하고 t-SNE로 시각화합니다.

    python evaluate.py

---

## 4. 코드 파일 구성 및 설명

* **`args.py`**: Hyperparameters, LoRA, Model 설정값을 정의하는 데이터 클래스 모음입니다.
* **`crush_loss.py`**: K-Means 중심점을 관리하는 `CentroidManager`와 마진 기반의 3가지 CRUSH Loss 함수를 계산하는 핵심 로직입니다.
* **`dataset.py`**: 안전/유해 데이터셋을 API 및 로컬 파일에서 로드하고, Seeded K-Means를 통해 유해 프롬프트를 5개 카테고리로 사전 라벨링하는 전처리를 수행합니다.
* **`evaluate.py`**: 학습된 어댑터를 적용하여 벡터를 추출하고, 클러스터링 평가 지표 계산 및 t-SNE 시각화를 수행합니다.
* **`train.py`**: 환경 변수 설정, LoRA 설정, 데이터셋 및 `CustomTrainer`를 초기화하여 실제 학습 루프를 시작하는 진입점(Entry point)입니다.
* **`trainer.py`**: `SFTTrainer`를 상속받아 `_compute_loss`를 오버라이딩합니다. Before/After 모델의 Representation을 비교하고, Step별로 디버깅 로그를 파일로 기록합니다.
* **`utils.py`**: Online sample 생성을 위한 생성 텍스트 후처리 및 Stopping Criteria 등 편의 기능을 제공합니다.

---

## 5. 데이터셋 및 전처리 과정 (`dataset.py`)

CRUSH 프로젝트에서 모델이 '안전함'과 '유용함'의 균형을 맞추기 위해 사용하는 3가지 핵심 데이터셋에 대해 설명합니다. 이 프로젝트의 `dataset.py`는 세 가지 각기 다른 특성을 가진 데이터셋을 결합하여 모델의 잠재 공간(Latent Space)을 정교하게 조율합니다.

### 5.1 UltraChat 200k (`HuggingFaceH4/ultrachat_200k`)
* **데이터셋 자체의 특징**: ChatGPT를 활용해 생성된 대규모 고품질 합성 대화 데이터셋입니다. 일상적인 질문, 정보성 대화, 코딩 질문 등 매우 광범위하고 다양한 주제의 지시(Instruction)와 그에 대한 정상적이고 유용한 응답을 포함하고 있습니다.
* **CRUSH 코드에서의 역할**:
  * **목적**: 모델이 유해한 질문을 차단하는 방법을 배우더라도, 일반적인 대화 능력이나 유용성을 잃지 않도록 돕는 '안전(Safe) 베이스라인' 역할을 합니다.
  * **처리 방식**: 코드 내에서 HuggingFace API를 통해 500개의 무해한 프롬프트와 무해한 응답(harmless prompt + harmless answer) 쌍을 로드하여 `data_safe_samples` 리스트에 추가합니다.

### 5.2 WildJailbreak (`allenai/wildjailbreak`)
* **데이터셋 자체의 특징**: AllenAI에서 제작한 대규모 오픈소스 안전성 훈련 데이터셋입니다. 단순한 유해 요청뿐만 아니라, 모델의 보안을 우회하려는 복잡한 '탈옥(Jailbreak)' 프롬프트들을 포함하고 있습니다. 가장 큰 특징은 유해한 의도가 없지만 위험해 보이는 프롬프트(Vanilla Benign)를 포함한다는 점입니다. 이를 통해 모델이 무조건적으로 답변을 거부하는 '과도한 안전성(Over-refusal)'에 빠지지 않도록 방지합니다.
* **CRUSH 코드에서의 역할**:
  * **목적**: 노골적인 유해 프롬프트와 정상 프롬프트의 미세한 경계를 모델에게 학습시키고, 과도한 거부를 방지합니다.
  * **처리 방식**: 로컬의 `wildjailbreak.jsonl` 파일에서 데이터를 읽어옵니다.
    * **안전 데이터(250개)**: `harmless` 프롬프트를 추출하여 UltraChat과 함께 `data_safe_samples`에 병합합니다.
    * **위험 쌍(250개)**: `harmful` 프롬프트에 대해 모델이 뱉은 유해한 답변(harmful answer)과 안전하게 거부한 답변(harmless answer)을 짝지어 `unsafe_prompt_pair_unsafe_answer` 및 `unsafe_prompt_pair_safe_answer`로 저장합니다.

### 5.3 Do-Not-Answer (`LibrAI/do-not-answer`)
* **데이터셋 자체의 특징**: LLM이 절대 대답해서는 안 되는 오픈소스 악성 프롬프트 데이터셋입니다. 범죄 모의, 정보 유출, 혐오 발언 등 명확하게 위험한 질문들로만 구성되어 있으며, 각 프롬프트는 5가지 주요 위험 범주(Risk Area)로 분류되어 있습니다.
* **CRUSH 코드에서의 역할**:
  * **목적**: CRUSH 방법론의 핵심인 '유해성 카테고리별 클러스터링'을 위한 타겟 데이터로 사용됩니다.
  * **처리 방식**:
    1. HuggingFace API를 통해 500개의 유해 프롬프트와 기존 모델들(GPT-4, ChatGPT 등)이 생성했던 유해한 응답을 추출합니다.
    2. 데이터 로드 중 다음 5가지 카테고리에 해당하는 프롬프트를 정확히 1개씩만 추출하여 시드(Seed)로 저장합니다.
       * Malicious Uses (범죄 악용)
       * Human-Chatbot Interaction Harms (정서적 가스라이팅/의존)
       * Information Hazards (개인/위험 정보 유출)
       * Misinformation Harms (허위 사실/음모론)
       * Discrimination, Exclusion, Toxicity, Hateful, Offensive (차별 및 혐오)

### 5.4 Seeded K-Means 클러스터링 및 토크나이징
* **클러스터링**: 위에서 추출된 5개의 시드 프롬프트들을 초기 중심점으로 삼아 Seeded K-Means 클러스터링(TF-IDF 벡터라이저 사용)을 수행합니다. 이를 통해 나머지 유해 프롬프트들이 어떤 범주에 속하는지 라벨(0~4)을 부여하고 `data_unsafe_samples`로 저장합니다.
* **토크나이징**: 모델 템플릿(Qwen, Llama 등)에 맞춰 `<SEPARATOR>`를 기준으로 요청과 응답을 나누어 텐서로 변환합니다.

> 💡 **요약하자면**:
> CRUSH는 Do-Not-Answer를 통해 유해한 요청들을 5개의 뚜렷한 군집으로 묶어서 밀어내고(Push/Pull Loss), 동시에 UltraChat과 WildJailbreak의 안전한 데이터들을 이 위험 군집들로부터 멀리 떼어놓음(Benign Loss)으로써 언어 모델의 '지식 공간'을 안전하게 재배치하는 것입니다.

---

## 6. 평가 과정 및 기준 (`evaluate.py`)

평가는 크게 정성적 생성 확인과 정량적 공간 분석으로 나뉩니다.

* **예측 테스트 (Inference)**: 테스트 프롬프트를 입력하고 마지막 토큰의 은닉 상태(Hidden State)를 추출한 뒤, `crush_centroids.pt`에 저장된 5개 중심점과의 유클리디안 거리를 계산하여 가장 가까운 카테고리로 분류합니다.
* **공간 분리도 지표**: Silhouette Score (> 0.3 양호, > 0.5 우수) 및 Davies-Bouldin Score (낮을수록 좋음)를 측정하여 LoRA 적용 전/후의 클러스터링 개선도를 확인합니다.
* **시각화 (t-SNE)**: 5개 카테고리로 묶인 벡터들의 분포를 Before/After 2D 산점도로 시각화하여 밀집도를 직관적으로 검증합니다.

---

## 7. 학습 과정 상세 설명 (`train.py`, `trainer.py`)

학습은 기존 모델 가중치는 동결한 채 LoRA 가중치만 업데이트하는 파인튜닝 방식으로 진행됩니다.

* **자동 레이어 타겟팅**: `train.py`에서 모델의 구조(Architecture)를 파악해, 총 레이어 수의 후반부 50%를 LoRA 적용 및 Representation 타겟 레이어로 자동 설정합니다.
* **은닉 상태(Hidden State) 추출**: `trainer.py`의 `_get_org_model_repr` 함수에서 LoRA 어댑터를 비활성화(`disable_adapter`)하여 Base 모델의 원래 벡터를 추출하고, 다시 활성화하여 현재 스텝의 변형된 벡터를 추출합니다.
* **중심점(Centroid) 이동**: 매 Step마다 현재 배치의 Unsafe 벡터($h_{un\_new}$) 평균을 구해, `CentroidManager`가 Momentum(0.9) 방식으로 중심점 좌표를 부드럽게 업데이트합니다 (그래디언트는 차단).
* **손실 계산 및 디버깅**: 계산된 4가지 Loss(Benign, Pull, Push, KL) 값을 통해 Backpropagation을 수행합니다. 동시에 DebugLogger를 통해 Step마다 각 Loss의 변화량, 라벨 분포, 중심점 간의 평균 거리를 `./debug_training_log.txt`에 상세히 기록합니다.

---

## 8. 기타 부가 설명

* **4비트 양자화 (BitsAndBytes)**: 제한된 VRAM 환경에서도 대형 모델을 훈련하기 위해 `nf4` 타입의 4-bit 양자화(Double Quantization 적용)를 활성화하여 로드합니다.
* **Gradient Checkpointing 활성화**: VRAM 효율을 극대화하기 위해 `training_args.gradient_checkpointing`이 켜져 있을 때 `enable_input_require_grads()`를 통해 그래디언트 흐름을 유지합니다.
* **동적 시드 고정**: 재현성을 위해 CUDA 뿐만 아니라 Apple Silicon(MPS) 환경에 대해서도 하드웨어에 맞게 시드를 고정하는 로직이 적용되어 있습니다.