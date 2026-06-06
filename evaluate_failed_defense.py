"""
evaluate_failed_defense.py
==========================
failed_defense_augmented_dna.jsonl에 대해 3가지 응답 비교:

  (A) base    : LoRA 적용 안 한 base 모델 답변
  (B) lora    : LoRA만 적용한 답변 (카테고리 제어 없음)
  (C) crush   : LoRA 분류 → 카테고리 기반 응답 제어 (CRUSH 컨트롤러)

추가로:
  - 분류 정확도 (true risk_area vs 예측)
  - 카테고리별 confusion matrix
  - 결과 jsonl 저장 (사람이 직접 비교 가능)

사용:
  python evaluate_failed_defense.py
  python evaluate_failed_defense.py --limit 50  # 처음 50개만 (빠른 검증)
"""

import os
import json
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경 대응
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from inference_controller import CRUSHController
from cluster_label_mapping import RISK_NAME_MAP


# 데이터셋의 risk_area 문자열 → 0~4 인덱스
RISK_AREA_TO_IDX = {
    "Malicious Uses": 0,
    "Human-Chatbot Interaction Harms": 1,
    "Information Hazards": 2,
    "Misinformation Harms": 3,
    "Discrimination, Exclusion, Toxicity, Hateful, Offensive": 4,
}


def load_dataset(jsonl_path):
    """failed_defense_augmented_dna.jsonl 로드."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            risk = d.get("risk_area", "")
            if risk not in RISK_AREA_TO_IDX:
                continue  # 알 수 없는 카테고리 스킵
            data.append({
                "question": d["question"],
                "risk_area": risk,
                "risk_idx": RISK_AREA_TO_IDX[risk],
                "reference_response": d.get("response", ""),  # 평가용 reference
                "base_response_ref": d.get("base_response", ""),  # 데이터셋이 미리 만든 base 응답
                "source": "failed_defense",  # 데이터 출처 (시각화 시 모양 구분)
            })
    return data


def load_dna_samples(base_model_name, num_per_class=50):
    """
    Do-Not-Answer에서 카테고리별 균등 추출 (failed_defense와 시각화 함께 비교용).

    Args:
        base_model_name: dataset.py가 템플릿 선택할 때 사용
        num_per_class: 카테고리별 가져올 샘플 수
    """
    print(f"\n📥 Do-Not-Answer 학습 데이터 로드 (카테고리별 {num_per_class}개)...")
    from transformers import AutoTokenizer
    from dataset import RepBendingDataset

    dummy_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dummy_tokenizer.pad_token = dummy_tokenizer.eos_token or dummy_tokenizer.unk_token

    train_dataset = RepBendingDataset(
        tokenizer=dummy_tokenizer,
        num_examples=500,
        mode="response_all",
        max_length=512,
        model_name_or_path=base_model_name,
        dataset_path="LibrAI/do-not-answer",
        split="train",
        is_online=False,
    )

    data = []
    true_arr = np.array(train_dataset.data_unsafe_true_labels)
    for label_idx in range(5):
        indices = np.where(true_arr == label_idx)[0][:num_per_class]
        for idx in indices:
            prompt = train_dataset.data_unsafe_request_prompts[idx]
            risk_name_full = [k for k, v in RISK_AREA_TO_IDX.items() if v == label_idx][0]
            data.append({
                "question": prompt,
                "risk_area": risk_name_full,
                "risk_idx": label_idx,
                "reference_response": "",
                "base_response_ref": "",
                "source": "do_not_answer",
            })

    bincount = np.bincount([d["risk_idx"] for d in data], minlength=5)
    print(f"  DNA 로드 완료: {len(data)}개, 분포={bincount.tolist()}")
    return data


def evaluate(
    controller: CRUSHController,
    data: list,
    output_path: str,
    limit: int = None,
    skip_generation: bool = False,
):
    """
    각 샘플에 대해 분류 + 3가지 응답 생성. 결과를 output_path에 jsonl로 저장.

    Args:
        skip_generation: True면 A/B 생성 건너뛰고 분류 + C(crush)만 수행 (빠른 테스트).
    """
    if limit:
        data = data[:limit]

    n = len(data)
    print(f"\n📊 평가 시작: {n}개 샘플")
    if skip_generation:
        print("  ⚠️ skip_generation=True → A(base)/B(lora) 답변 생성 안 함, 분류 + C(crush)만 수행")

    results = []
    start = time.time()

    # 분류 통계용
    pred_indices = []
    true_indices = []
    # 시각화용 hidden vector 수집 (Before/After 둘 다)
    hidden_after = []   # LoRA ON
    hidden_before = []  # LoRA OFF (base 모델)
    sources = []        # 데이터 출처 (시각화 시 모양 구분용)

    for i, sample in enumerate(data):
        prompt = sample["question"]
        true_risk = sample["risk_area"]
        true_idx = sample["risk_idx"]
        source = sample.get("source", "failed_defense")

        # ----- C: CRUSH controller (분류 + 응답 제어, hidden 수집) -----
        crush_result = controller.generate_controlled(prompt, return_hidden=True)
        cls = crush_result["classification"]
        pred_idx = cls["predicted_pseudo"]
        pred_risk_name = cls["predicted_risk_name"]
        pred_indices.append(pred_idx)
        true_indices.append(true_idx)
        hidden_after.append(cls["hidden"])
        sources.append(source)

        # ----- Before LoRA hidden 추출 (LoRA OFF) -----
        h_before = controller.extract_hidden_before(prompt)
        hidden_before.append(h_before)

        record = {
            "idx": i,
            "question": prompt,
            "source": source,
            "true_risk_area": true_risk,
            "true_risk_idx": true_idx,
            "predicted_pseudo": pred_idx,
            "predicted_risk_name": pred_risk_name,
            "distances_to_centroids": cls["distances"],
            "min_distance": cls["min_distance"],
            "confidence_gap": cls["confidence_gap"],
            "C_strategy": crush_result["strategy"],
            "C_crush_response": crush_result["response"],
        }

        # ----- A: base 답변 -----
        if not skip_generation:
            try:
                base_resp = controller.generate_base(prompt, max_new_tokens=200)
                record["A_base_response"] = base_resp
            except Exception as e:
                record["A_base_response"] = f"<ERROR: {e}>"

            # ----- B: LoRA only 답변 -----
            try:
                lora_resp = controller.generate_lora_only(prompt, max_new_tokens=200)
                record["B_lora_response"] = lora_resp
            except Exception as e:
                record["B_lora_response"] = f"<ERROR: {e}>"

        results.append(record)

        # 진행상황
        if (i + 1) % 10 == 0 or i == n - 1:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  [{i + 1}/{n}] elapsed={elapsed:.1f}s, ETA={eta:.1f}s, "
                  f"true={true_risk[:20]} → pred={pred_risk_name}")

    # ---------- 저장 ----------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n💾 결과 저장: {output_path}")

    # ---------- 분류 정확도 ----------
    print_classification_report(pred_indices, true_indices, controller)

    # ---------- t-SNE 시각화 (Before vs After × TRUE label) ----------
    hidden_after_arr = np.stack(hidden_after, axis=0)
    hidden_before_arr = np.stack(hidden_before, axis=0)
    centroids_np = controller.centroids.detach().cpu().numpy()
    pseudo_to_risk_idx = controller.mapping["pseudo_to_risk_idx"]

    plot_path = output_path.replace(".jsonl", "_tsne.png")
    visualize_before_after_tsne(
        hidden_before=hidden_before_arr,
        hidden_after=hidden_after_arr,
        true_labels=np.array(true_indices),
        sources=np.array(sources),
        centroids=centroids_np,
        pseudo_to_risk_idx=pseudo_to_risk_idx,
        cluster_layer=controller.cluster_layer,
        save_path=plot_path,
    )

    return results


# =========================================================
# t-SNE 시각화: Before vs After LoRA (TRUE label로 색칠)
# =========================================================
def visualize_before_after_tsne(
    hidden_before,
    hidden_after,
    true_labels,
    centroids,
    pseudo_to_risk_idx,
    cluster_layer,
    save_path,
    sources=None,
):
    """
    Before LoRA (base 모델) vs After LoRA (LoRA 적용) hidden을 같은 t-SNE 공간에 투영하여
    학습 효과를 시각적으로 비교.

    색칠: TRUE risk_area (정답 카테고리, 5색)
    모양: 데이터 출처 (sources 인자가 있으면)
      - "do_not_answer": 원 (●)
      - "failed_defense": 세모 (▲)
      - sources=None이면 모두 원으로 표시
    Centroid: 별 (★, After 그림에만)
    """
    print(f"\n🎨 t-SNE 시각화 생성 중... (cluster_layer={cluster_layer}, N={len(hidden_after)})")

    n_before = len(hidden_before)
    n_after = len(hidden_after)
    n_cent = len(centroids)

    # ★ Before와 After를 따로 t-SNE (원본 evaluate.py 방식)
    # 같은 공간에 넣으면 t-SNE가 두 분포를 동시에 표현하려고 타협해서 학습 효과가 흐려짐.
    # 각자 독립된 t-SNE 공간으로 가야 응집/분리 변화가 명확히 보임.
    # Centroid는 After 공간과 함께 투영 (After에만 그림).
    perp_before = min(30, max(5, n_before // 10))
    perp_after = min(30, max(5, n_after // 10))

    print(f"  Before t-SNE 실행 (perp={perp_before})...")
    tsne_b = TSNE(n_components=2, random_state=42, perplexity=perp_before, init="pca")
    emb_before = tsne_b.fit_transform(hidden_before)

    print(f"  After + Centroids t-SNE 실행 (perp={perp_after})...")
    after_combined = np.vstack([hidden_after, centroids])
    tsne_a = TSNE(n_components=2, random_state=42, perplexity=perp_after, init="pca")
    emb_after_combined = tsne_a.fit_transform(after_combined)
    emb_after = emb_after_combined[:n_after]
    emb_centroids = emb_after_combined[n_after:]

    colors = {
        0: "#e74c3c",  # Malicious
        1: "#3498db",  # ChatbotHarm
        2: "#2ecc71",  # InfoHazard
        3: "#f39c12",  # Misinfo
        4: "#9b59b6",  # Discrim
    }
    # 데이터셋별 모양
    source_markers = {
        "do_not_answer": "o",      # 원
        "failed_defense": "^",     # 세모
    }

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    title_suffix = " (color=category, shape=dataset)" if sources is not None else ""
    fig.suptitle(
        f"CRUSH: Latent Space Before vs After LoRA (layer={cluster_layer}, TRUE risk_area){title_suffix}",
        fontsize=14, fontweight="bold",
    )

    for ax, emb, title, show_centroids in zip(
        axes,
        [emb_before, emb_after],
        ["Before LoRA (base model)", "After LoRA (CRUSH applied)"],
        [False, True],
    ):
        # 점 찍기
        if sources is not None:
            unique_sources = sorted(set(sources.tolist()))
            for src in unique_sources:
                marker = source_markers.get(src, "o")
                src_mask = sources == src
                for risk_i in range(5):
                    mask = src_mask & (true_labels == risk_i)
                    if mask.sum() == 0:
                        continue
                    ax.scatter(
                        emb[mask, 0], emb[mask, 1],
                        c=colors[risk_i], alpha=0.6, s=50,
                        marker=marker, edgecolors="white", linewidths=0.5,
                        label=f"{RISK_NAME_MAP[risk_i]} / {src} ({mask.sum()})",
                    )
        else:
            for risk_i in range(5):
                mask = true_labels == risk_i
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=colors[risk_i], alpha=0.6, s=45,
                    label=f"{RISK_NAME_MAP[risk_i]} ({mask.sum()})",
                )

        # After 그림에만 centroid 별표
        if show_centroids:
            for pseudo_i in range(len(emb_centroids)):
                risk_i = pseudo_to_risk_idx.get(pseudo_i, -1)
                ec = colors.get(risk_i, "black")
                ax.scatter(
                    emb_centroids[pseudo_i, 0], emb_centroids[pseudo_i, 1],
                    c="black", marker="*", s=500, edgecolors=ec, linewidths=2.5, zorder=10,
                )
                risk_name = RISK_NAME_MAP.get(risk_i, f"C{pseudo_i}")
                ax.annotate(
                    f"★{risk_name}",
                    (emb_centroids[pseudo_i, 0], emb_centroids[pseudo_i, 1]),
                    fontsize=10, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ec, alpha=0.85),
                )

        ax.set_title(title, fontsize=12)
        ax.legend(loc="best", fontsize=7, framealpha=0.85, ncol=2 if sources is not None else 1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 t-SNE Before/After 그림 저장: {save_path}")

    # 정량 metric: 카테고리별 응집도 변화
    print_separation_metrics(hidden_before, hidden_after, true_labels)

    # 추가로 거리 분포 히스토그램 (After 기준)
    hist_path = save_path.replace("_tsne.png", "_confidence_hist.png")
    plot_confidence_histogram(hidden_after, centroids, true_labels, hist_path)


def print_separation_metrics(hidden_before, hidden_after, true_labels):
    """
    카테고리별 응집도(intra-cluster distance)와 분리도(inter-cluster distance)
    Before vs After 정량 비교.

    학습이 성공했다면:
      - intra(같은 카테고리 내 평균 거리) ↓
      - inter(다른 카테고리 간 평균 거리) ↑
    """
    from sklearn.metrics.pairwise import euclidean_distances

    print("\n" + "=" * 60)
    print("📐 정량 분리도 측정 (TRUE label 기준)")
    print("=" * 60)

    for label, vectors in [("Before", hidden_before), ("After", hidden_after)]:
        print(f"\n[{label} LoRA]")

        # intra
        intra_total = []
        for risk_i in range(5):
            mask = true_labels == risk_i
            if mask.sum() < 2:
                continue
            cluster_vecs = vectors[mask]
            dists = euclidean_distances(cluster_vecs)
            np.fill_diagonal(dists, np.nan)
            intra = np.nanmean(dists)
            intra_total.append(intra)
            print(f"  intra[{RISK_NAME_MAP[risk_i]:<12}]: {intra:.4f}  (n={mask.sum()})")
        if intra_total:
            print(f"  intra 평균: {np.mean(intra_total):.4f}  ← 작을수록 응집 좋음")

        # inter (카테고리별 mean point 간 거리)
        mean_points = []
        labels_present = []
        for risk_i in range(5):
            mask = true_labels == risk_i
            if mask.sum() > 0:
                mean_points.append(vectors[mask].mean(axis=0))
                labels_present.append(risk_i)
        if len(mean_points) >= 2:
            mean_points = np.array(mean_points)
            inter_dists = euclidean_distances(mean_points)
            inter = []
            for i in range(len(mean_points)):
                for j in range(i + 1, len(mean_points)):
                    inter.append(inter_dists[i, j])
            print(f"  inter 평균: {np.mean(inter):.4f}  ← 클수록 분리 좋음")
    print("=" * 60)


def plot_confidence_histogram(hidden_vectors, centroids, true_labels, save_path):
    """
    각 샘플의 (1st-nearest centroid 거리) vs (2nd-nearest centroid 거리)
    차이(=confidence gap) 분포를 카테고리별로 히스토그램으로.

    Gap이 클수록 분류기가 자신감 있게 결정한 것.
    카테고리별로 어디서 자신감이 떨어지는지 진단 가능.
    """
    import torch
    h = torch.from_numpy(hidden_vectors).float()
    c = torch.from_numpy(centroids).float()
    dists = torch.cdist(h, c)  # (N, K)
    sorted_d, _ = torch.sort(dists, dim=1)
    gaps = (sorted_d[:, 1] - sorted_d[:, 0]).numpy()
    min_dists = sorted_d[:, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 왼쪽: confidence gap 히스토그램 (카테고리별)
    ax = axes[0]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for risk_i in range(5):
        mask = true_labels == risk_i
        if mask.sum() == 0:
            continue
        ax.hist(gaps[mask], bins=20, alpha=0.5, color=colors[risk_i],
                label=f"{RISK_NAME_MAP[risk_i]} ({mask.sum()})")
    ax.set_xlabel("Confidence gap (2nd - 1st nearest distance)")
    ax.set_ylabel("Count")
    ax.set_title("Classifier confidence by true category\n(higher = more confident)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 오른쪽: 최단 centroid 거리 (카테고리별)
    ax = axes[1]
    for risk_i in range(5):
        mask = true_labels == risk_i
        if mask.sum() == 0:
            continue
        ax.hist(min_dists[mask], bins=20, alpha=0.5, color=colors[risk_i],
                label=f"{RISK_NAME_MAP[risk_i]} ({mask.sum()})")
    ax.set_xlabel("Distance to nearest centroid")
    ax.set_ylabel("Count")
    ax.set_title("Closeness to nearest centroid by true category\n(lower = better)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 confidence 히스토그램 저장: {save_path}")


def print_classification_report(pred_indices, true_indices, controller):
    """분류 정확도 + confusion matrix 출력."""
    print("\n" + "=" * 60)
    print("📈 분류 결과 리포트")
    print("=" * 60)

    pred_arr = np.array(pred_indices)
    true_arr = np.array(true_indices)
    n = len(pred_arr)

    # pseudo → risk_idx 매핑 적용해서 같은 공간에서 비교
    pseudo_to_risk_idx = controller.mapping["pseudo_to_risk_idx"]
    predicted_risk_idx = np.array([
        pseudo_to_risk_idx.get(p, -1) for p in pred_indices
    ])

    correct = (predicted_risk_idx == true_arr).sum()
    acc = correct / n if n > 0 else 0.0
    print(f"\n전체 분류 정확도 (after Hungarian mapping): {acc:.4f}  ({correct}/{n})")

    # 카테고리별 정확도
    print("\n카테고리별 정확도:")
    for true_i in range(5):
        mask = (true_arr == true_i)
        if mask.sum() == 0:
            continue
        cat_acc = (predicted_risk_idx[mask] == true_i).sum() / mask.sum()
        name = RISK_NAME_MAP[true_i]
        print(f"  {name:<15}: {cat_acc:.4f}  ({(predicted_risk_idx[mask] == true_i).sum()}/{mask.sum()})")

    # confusion matrix (true × predicted-risk-name)
    print("\nConfusion Matrix (행=true risk, 열=predicted risk):")
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(true_arr, predicted_risk_idx):
        if 0 <= p < 5:
            cm[t, p] += 1

    header = "         " + "  ".join([f"  {RISK_NAME_MAP[j][:5]:<5}" for j in range(5)])
    print(header)
    for i in range(5):
        row = "  ".join([f"{cm[i, j]:5d}" for j in range(5)])
        print(f"  {RISK_NAME_MAP[i][:10]:<10} {row}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--lora_path", default="./out/test")
    parser.add_argument("--centroids_path", default="./out/test/crush_centroids.pt")
    parser.add_argument("--init_centroids_path", default="./out/test/crush_centroids_initial.pt")
    parser.add_argument("--mapping_path", default="./out/test/cluster_label_mapping.json")
    parser.add_argument("--data_path", default="./failed_defense_augmented_dna.jsonl")
    parser.add_argument("--output_path", default="./out/test/failed_defense_results.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="처음 N개만 평가 (디버깅용)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="A/B 답변 생성 건너뛰고 분류 + C만 (빠른 검증)")
    parser.add_argument("--no_4bit", action="store_true", help="4bit 양자화 끄기")
    parser.add_argument("--include_dna", action="store_true",
                        help="Do-Not-Answer 데이터도 같이 평가 (시각화에서 모양으로 구분)")
    parser.add_argument("--dna_per_class", type=int, default=50,
                        help="DNA에서 카테고리별 가져올 샘플 수 (--include_dna일 때만)")
    args = parser.parse_args()

    # mapping 없으면 자동 생성
    if not os.path.exists(args.mapping_path):
        print(f"⚠️ {args.mapping_path} 없음. cluster_label_mapping.py로 자동 생성 시도...")
        from cluster_label_mapping import create_mapping_from_centroid_file
        out_dir = os.path.dirname(args.mapping_path)
        create_mapping_from_centroid_file(args.init_centroids_path, out_dir)

    # 컨트롤러 초기화
    controller = CRUSHController(
        base_model_name=args.base_model,
        lora_path=args.lora_path,
        centroids_path=args.centroids_path,
        mapping_path=args.mapping_path,
        init_centroid_meta_path=args.init_centroids_path,
        load_in_4bit=not args.no_4bit,
    )

    # 데이터 로드
    print(f"\n📥 데이터 로드: {args.data_path}")
    data = load_dataset(args.data_path)
    print(f"  failed_defense {len(data)}개 샘플")

    # DNA 합치기 (옵션)
    if args.include_dna:
        dna_data = load_dna_samples(args.base_model, num_per_class=args.dna_per_class)
        data = dna_data + data
        print(f"  ▶ 통합: DNA {len(dna_data)}개 + failed_defense {len(data) - len(dna_data)}개 "
              f"= 총 {len(data)}개")

    # 평가 실행
    evaluate(
        controller=controller,
        data=data,
        output_path=args.output_path,
        limit=args.limit,
        skip_generation=args.skip_generation,
    )


if __name__ == "__main__":
    main()