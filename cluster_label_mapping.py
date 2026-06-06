"""
cluster_label_mapping.py
========================
학습 후 pseudo-label (0~4) ↔ risk_area (Malicious/ChatbotHarm/...) 매핑 생성.

배경:
  - K-Means가 만든 pseudo-label 0~4는 임의 인덱스라 의미가 없음
  - Inference 시 "C2 클러스터에 가깝다"는 정보를 받아도 "이게 Malicious인지 Misinfo인지" 모름
  - 학습 데이터의 (pseudo_labels, true_labels) 쌍에 Hungarian matching을 적용해
    "pseudo C0 ≈ Malicious", "pseudo C1 ≈ Discrim" 같은 매핑을 자동으로 찾음

저장:
  - {output_dir}/cluster_label_mapping.json
    - pseudo_to_risk: {"0": "Malicious", "1": "Discrim", ...}
    - risk_to_pseudo: 역방향
    - accuracy: Hungarian matching 후 분류 정확도
    - confusion_matrix: 학습 데이터에서 (true × pseudo) 분포
"""

import os
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# risk_area index → 이름 (DNA 정답 라벨)
RISK_NAME_MAP = {
    0: "Malicious",
    1: "ChatbotHarm",
    2: "InfoHazard",
    3: "Misinfo",
    4: "Discrim",
}

# 전체 이름 (사용자에게 표시할 때)
RISK_FULL_NAME_MAP = {
    0: "Malicious Uses",
    1: "Human-Chatbot Interaction Harms",
    2: "Information Hazards",
    3: "Misinformation Harms",
    4: "Discrimination, Exclusion, Toxicity, Hateful, Offensive",
}


def build_cluster_label_mapping(
    pseudo_labels,
    true_labels,
    num_classes=5,
):
    """
    pseudo_labels와 true_labels로부터 Hungarian matching 매핑 생성.
    
    Args:
        pseudo_labels: List[int] or np.ndarray, K-Means가 만든 클러스터 ID (0~K-1)
        true_labels: List[int] or np.ndarray, DNA의 risk_area 정답 (0~K-1)
        num_classes: 클래스 수 (5)
    
    Returns:
        mapping: dict with keys
            - pseudo_to_risk_idx: {pseudo_label(int): risk_area_idx(int)}
            - pseudo_to_risk_name: {pseudo_label(int): "Malicious", ...}
            - risk_idx_to_pseudo: {risk_area_idx(int): pseudo_label(int)}
            - confusion_matrix: (true x pseudo) 2D list
            - hungarian_accuracy: float
    """
    pseudo_arr = np.asarray(pseudo_labels)
    true_arr = np.asarray(true_labels)

    # confusion matrix: cm[true, pseudo] = count
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_arr, pseudo_arr):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Hungarian matching (maximize → -cm으로 minimize)
    row_ind, col_ind = linear_sum_assignment(-cm)
    # row_ind[i] = true class, col_ind[i] = matched pseudo class

    pseudo_to_risk_idx = {}
    risk_idx_to_pseudo = {}
    for true_i, pseudo_j in zip(row_ind, col_ind):
        pseudo_to_risk_idx[int(pseudo_j)] = int(true_i)
        risk_idx_to_pseudo[int(true_i)] = int(pseudo_j)

    pseudo_to_risk_name = {
        p: RISK_NAME_MAP[r] for p, r in pseudo_to_risk_idx.items()
    }

    matched_count = sum(cm[t, p] for t, p in zip(row_ind, col_ind))
    accuracy = matched_count / len(true_arr) if len(true_arr) > 0 else 0.0

    return {
        "pseudo_to_risk_idx": pseudo_to_risk_idx,
        "pseudo_to_risk_name": pseudo_to_risk_name,
        "risk_idx_to_pseudo": risk_idx_to_pseudo,
        "confusion_matrix": cm.tolist(),
        "hungarian_accuracy": float(accuracy),
        "num_classes": num_classes,
        "risk_name_map": RISK_NAME_MAP,
        "risk_full_name_map": RISK_FULL_NAME_MAP,
    }


def save_mapping(mapping, output_path):
    """JSON으로 저장. 키는 모두 string으로 변환 (JSON 호환)."""
    serializable = {
        "pseudo_to_risk_idx": {str(k): v for k, v in mapping["pseudo_to_risk_idx"].items()},
        "pseudo_to_risk_name": {str(k): v for k, v in mapping["pseudo_to_risk_name"].items()},
        "risk_idx_to_pseudo": {str(k): v for k, v in mapping["risk_idx_to_pseudo"].items()},
        "confusion_matrix": mapping["confusion_matrix"],
        "hungarian_accuracy": mapping["hungarian_accuracy"],
        "num_classes": mapping["num_classes"],
        "risk_name_map": {str(k): v for k, v in mapping["risk_name_map"].items()},
        "risk_full_name_map": {str(k): v for k, v in mapping["risk_full_name_map"].items()},
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"💾 cluster label mapping 저장: {output_path}")


def load_mapping(mapping_path):
    """저장된 매핑을 로드하고 키를 int로 복원."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "pseudo_to_risk_idx": {int(k): v for k, v in data["pseudo_to_risk_idx"].items()},
        "pseudo_to_risk_name": {int(k): v for k, v in data["pseudo_to_risk_name"].items()},
        "risk_idx_to_pseudo": {int(k): v for k, v in data["risk_idx_to_pseudo"].items()},
        "confusion_matrix": data["confusion_matrix"],
        "hungarian_accuracy": data["hungarian_accuracy"],
        "num_classes": data["num_classes"],
        "risk_name_map": {int(k): v for k, v in data["risk_name_map"].items()},
        "risk_full_name_map": {int(k): v for k, v in data["risk_full_name_map"].items()},
    }


def print_mapping_report(mapping):
    """매핑 결과를 보기 좋게 출력."""
    print("\n" + "=" * 60)
    print("📊 Cluster ↔ Risk Area 매핑 리포트")
    print("=" * 60)
    print(f"\nHungarian Accuracy: {mapping['hungarian_accuracy']:.4f}")
    print("\nPseudo Cluster → Risk Area:")
    for p, name in sorted(mapping["pseudo_to_risk_name"].items()):
        print(f"  Cluster {p} → {name}")

    print("\nConfusion Matrix (행=true, 열=pseudo):")
    cm = np.array(mapping["confusion_matrix"])
    header = "         " + "  ".join([f"  C{j}" for j in range(cm.shape[1])])
    print(header)
    for i in range(cm.shape[0]):
        row = "  ".join([f"{cm[i, j]:4d}" for j in range(cm.shape[1])])
        name = RISK_NAME_MAP.get(i, f"C{i}")[:10]
        print(f"  {name:<10} {row}")
    print("=" * 60)


def create_mapping_from_centroid_file(centroid_file_path, output_dir):
    """
    train.py가 저장한 crush_centroids_initial.pt에서 매핑 생성.
    
    Args:
        centroid_file_path: train.py가 저장한 'crush_centroids_initial.pt' 경로
        output_dir: 매핑 저장할 디렉토리
    """
    data = torch.load(centroid_file_path, map_location="cpu")

    if not isinstance(data, dict) or "pseudo_labels" not in data or "true_labels" not in data:
        raise ValueError(
            f"centroid_file_path={centroid_file_path}에 pseudo_labels/true_labels가 없음. "
            "train.py가 새 형식으로 저장한 파일이 아닐 수 있음."
        )

    pseudo_labels = data["pseudo_labels"]
    true_labels = data["true_labels"]
    num_classes = data.get("num_clusters", 5)

    print(f"📥 매핑 생성: pseudo={len(pseudo_labels)}개, true={len(true_labels)}개")

    mapping = build_cluster_label_mapping(pseudo_labels, true_labels, num_classes)
    print_mapping_report(mapping)

    output_path = os.path.join(output_dir, "cluster_label_mapping.json")
    save_mapping(mapping, output_path)
    return mapping


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        centroid_path = sys.argv[1]
        out_dir = os.path.dirname(centroid_path)
        create_mapping_from_centroid_file(centroid_path, out_dir)
    else:
        # 기본 경로 시도
        default_path = "./out/test/crush_centroids_initial.pt"
        if os.path.exists(default_path):
            create_mapping_from_centroid_file(default_path, os.path.dirname(default_path))
        else:
            print(f"사용법: python cluster_label_mapping.py <crush_centroids_initial.pt 경로>")
            print(f"또는 {default_path}에 파일을 두고 인자 없이 실행")
