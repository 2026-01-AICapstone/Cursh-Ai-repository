"""
inference_controller.py
=======================
CRUSH inference 컨트롤러.

핵심 아이디어:
  - LoRA는 "latent 분류기" 역할만 (어느 risk 카테고리에 가까운지 판별)
  - 답변 생성은 base 모델로 (LoRA 자체가 답변을 바꾸는 것 ≠ contribution)
  - 카테고리별로 다른 응답 전략 적용

사용 흐름:
  1. controller = CRUSHController(...)
  2. classification = controller.classify(prompt)
     → {predicted_risk_name, predicted_pseudo, distances, confidence}
  3. response = controller.generate(prompt)
     → 카테고리별 거부 메시지 + (필요시 카테고리 힌트 주입 후 base 답변)

비교군 (evaluate_failed_defense.py에서 사용):
  A) base 모델 단독 답변
  B) LoRA 모델 단독 답변  (LoRA가 답변에 미치는 영향만)
  C) CRUSH 컨트롤러 답변 (LoRA 분류 + 카테고리 기반 제어)
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from cluster_label_mapping import load_mapping, RISK_NAME_MAP, RISK_FULL_NAME_MAP


# =========================================================
# 카테고리별 응답 전략 (권장값: 권장값 ❷에서 결정)
# =========================================================

# 거부형 응답: Malicious, InfoHazard, Misinfo, Discrim
# 정서지원형 응답: ChatbotHarm (모델이 직접 답변하되 카테고리 힌트 주입)

REFUSAL_TEMPLATES = {
    "Malicious": (
        "I can't help with that. The request appears to involve potentially harmful actions "
        "(e.g., violating someone's rights, illegal activities, or causing harm to others). "
        "If you have a different question or need help with a legitimate topic, I'm happy to assist."
    ),
    "InfoHazard": (
        "I'm not able to provide that information, as it could be used to cause harm or "
        "compromise safety. If you have a related question for legitimate research, learning, "
        "or safety purposes, please rephrase and I'll do my best to help responsibly."
    ),
    "Misinfo": (
        "I can't confirm or generate that claim, as it appears to involve misinformation or "
        "unverified statements that could mislead others. I'd be glad to share what credible "
        "sources actually say about this topic."
    ),
    "Discrim": (
        "I won't engage with content that discriminates against, demeans, or generalizes about "
        "any group of people. If you have a constructive question about diversity, history, "
        "or social topics, I'm happy to discuss it respectfully."
    ),
    "ChatbotHarm": None,  # 모델 답변 사용 (아래 SYSTEM_HINT 참고)
}

# ChatbotHarm 카테고리 처리: base 모델에게 안전한 응답 가이드라인 제공 후 생성
CHATBOT_HARM_SYSTEM_HINT = (
    "The user may be in emotional distress or seeking inappropriate emotional dependency "
    "on an AI. Respond with empathy, acknowledge their feelings, but gently remind them that "
    "I am an AI assistant — not a substitute for human connection or professional support. "
    "If they appear to be in crisis, suggest they reach out to a trusted person or a "
    "mental health professional."
)


class CRUSHController:
    """
    CRUSH inference controller.
    
    구성 요소:
      - LoRA 모델 (분류용)
      - Base 모델 (답변 생성용) — LoRA를 disable_adapter()로 사용
      - centroids (5개)
      - cluster label mapping (pseudo → risk_name)
    """

    def __init__(
        self,
        base_model_name,
        lora_path,
        centroids_path,
        mapping_path,
        cluster_layer=None,  # None이면 init meta에서 읽음
        init_centroid_meta_path=None,  # crush_centroids_initial.pt
        load_in_4bit=True,
    ):
        print("=" * 60)
        print("🚀 CRUSHController 초기화")
        print("=" * 60)

        # ---------- Tokenizer ----------
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        # 생성용 padding은 left side (decoder-only)
        self.tokenizer.padding_side = "left"

        # ---------- Base 모델 + LoRA 어댑터 ----------
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, device_map="auto", quantization_config=bnb_config
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, device_map="auto", torch_dtype=torch.float16
            )

        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()
        self.device = self.model.device
        print(f"  ✅ 모델 로드: {base_model_name}, LoRA={lora_path}")

        # ---------- Centroids ----------
        centroids_raw = torch.load(centroids_path, map_location=self.device)
        if isinstance(centroids_raw, dict):
            centroids_raw = centroids_raw.get("centroids", centroids_raw)
        self.centroids = F.normalize(centroids_raw.float(), p=2, dim=-1).to(self.device)
        print(f"  ✅ centroids 로드: shape={self.centroids.shape}")

        # ---------- Cluster layer 결정 ----------
        if cluster_layer is None and init_centroid_meta_path:
            if os.path.exists(init_centroid_meta_path):
                meta = torch.load(init_centroid_meta_path, map_location="cpu")
                if isinstance(meta, dict) and "cluster_layer" in meta:
                    cluster_layer = meta["cluster_layer"]
                    print(f"  ✅ cluster_layer 자동 감지: {cluster_layer}")
        if cluster_layer is None:
            cluster_layer = -1
            print(f"  ⚠️ cluster_layer 미지정, 마지막 레이어(-1) 사용")
        self.cluster_layer = cluster_layer

        # ---------- Label mapping ----------
        self.mapping = load_mapping(mapping_path)
        self.pseudo_to_risk_name = self.mapping["pseudo_to_risk_name"]
        print(f"  ✅ mapping 로드: {self.pseudo_to_risk_name}")
        print("=" * 60)

    # =========================================================
    # Latent 분류
    # =========================================================
    @torch.no_grad()
    def classify(self, prompt: str, max_length: int = 256, return_hidden: bool = False):
        """
        프롬프트를 latent 공간에 던져 가장 가까운 centroid를 찾음.

        Args:
            prompt: 입력 프롬프트
            max_length: 토큰화 최대 길이
            return_hidden: True면 hidden vector (L2 normalized)도 함께 반환 (시각화용)

        Returns:
            {
                "predicted_pseudo": int (0~4),
                "predicted_risk_name": str,
                "distances": List[float] (5개 centroid까지의 거리),
                "min_distance": float,
                "confidence_gap": float (1st vs 2nd nearest 거리 차이),
                "hidden": np.ndarray (return_hidden=True인 경우만)
            }
        """
        # request만 사용 (response 없이)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)

        # LoRA 적용된 상태로 hidden 추출
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.cluster_layer]
        vec = hidden[0, -1, :].float()  # 마지막 토큰
        vec = F.normalize(vec, p=2, dim=-1)  # 학습/clustering과 동일

        # 5개 centroid와의 거리
        distances = torch.cdist(vec.unsqueeze(0), self.centroids).squeeze(0)

        sorted_dists, sorted_idx = torch.sort(distances)
        predicted_pseudo = int(sorted_idx[0].item())
        min_dist = float(sorted_dists[0].item())
        second_dist = float(sorted_dists[1].item()) if len(sorted_dists) > 1 else min_dist
        confidence_gap = second_dist - min_dist  # 클수록 자신감 ↑

        risk_name = self.pseudo_to_risk_name.get(predicted_pseudo, f"Unknown_{predicted_pseudo}")

        result = {
            "predicted_pseudo": predicted_pseudo,
            "predicted_risk_name": risk_name,
            "distances": distances.cpu().tolist(),
            "min_distance": min_dist,
            "confidence_gap": confidence_gap,
        }
        if return_hidden:
            result["hidden"] = vec.cpu().numpy()
        return result

    # =========================================================
    # Before LoRA hidden 추출 (LoRA OFF 상태)
    # =========================================================
    @torch.no_grad()
    def extract_hidden_before(self, prompt: str, max_length: int = 256):
        """
        LoRA 비활성화 상태(base 모델)에서 같은 cluster_layer의 hidden 추출.
        Before vs After 시각화 비교용.

        Returns:
            np.ndarray, L2 정규화된 hidden vector
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)

        # LoRA disable
        with self.model.disable_adapter():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[self.cluster_layer]
        vec = hidden[0, -1, :].float()
        vec = F.normalize(vec, p=2, dim=-1)
        return vec.cpu().numpy()

    # =========================================================
    # 카테고리 기반 응답 생성
    # =========================================================
    def generate_controlled(self, prompt: str, max_new_tokens: int = 200, return_hidden: bool = False):
        """
        분류 → 카테고리별 응답 전략 적용.

        Returns:
            {
                "classification": classify() 결과 (return_hidden=True면 hidden 포함),
                "strategy": "refusal" | "model_with_hint",
                "response": 최종 응답 텍스트,
            }
        """
        cls = self.classify(prompt, return_hidden=return_hidden)
        risk_name = cls["predicted_risk_name"]

        # 고정 거부 메시지 카테고리
        if risk_name in REFUSAL_TEMPLATES and REFUSAL_TEMPLATES[risk_name] is not None:
            response = REFUSAL_TEMPLATES[risk_name]
            return {
                "classification": cls,
                "strategy": "refusal",
                "response": response,
            }

        # ChatbotHarm: base 모델 답변 + system hint 주입
        if risk_name == "ChatbotHarm":
            response = self._generate_with_base(
                prompt,
                system_hint=CHATBOT_HARM_SYSTEM_HINT,
                max_new_tokens=max_new_tokens,
            )
            return {
                "classification": cls,
                "strategy": "model_with_hint",
                "response": response,
            }

        # 알 수 없는 카테고리 fallback
        return {
            "classification": cls,
            "strategy": "fallback",
            "response": "I'm not sure how to respond to this. Could you rephrase or clarify?",
        }

    # =========================================================
    # Base 모델 답변 (LoRA disable) — 비교군 A 및 ChatbotHarm 처리
    # =========================================================
    @torch.no_grad()
    def generate_base(self, prompt: str, max_new_tokens: int = 200):
        """LoRA 비활성화 상태로 base 모델 답변 생성."""
        with self.model.disable_adapter():
            return self._raw_generate(prompt, system_hint=None, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def generate_lora_only(self, prompt: str, max_new_tokens: int = 200):
        """LoRA 활성 상태로 답변 생성 (LoRA만 적용, 카테고리 제어 없음). 비교군 B."""
        return self._raw_generate(prompt, system_hint=None, max_new_tokens=max_new_tokens)

    def _generate_with_base(self, prompt: str, system_hint: str = None, max_new_tokens: int = 200):
        """ChatbotHarm 카테고리에서 system hint 포함하여 base 답변."""
        with self.model.disable_adapter():
            return self._raw_generate(prompt, system_hint=system_hint, max_new_tokens=max_new_tokens)

    def _raw_generate(self, prompt: str, system_hint: str = None, max_new_tokens: int = 200):
        """공통 generate 함수. apply_chat_template로 모델별 포맷 처리."""
        messages = []
        if system_hint:
            messages.append({"role": "system", "content": system_hint})
        messages.append({"role": "user", "content": prompt})

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # chat_template 없는 모델 fallback
            if system_hint:
                input_text = f"{system_hint}\n\nUser: {prompt}\nAssistant:"
            else:
                input_text = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 재현성을 위해 greedy
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # 입력 부분 제거
        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()