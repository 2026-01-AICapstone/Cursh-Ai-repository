import torch
import torch.nn.functional as F


class CentroidManager:
    def __init__(self, num_classes, hidden_size, momentum=0.9):
        self.num_classes = num_classes
        self.momentum = momentum
        # CPU에 두고 grad 완전 차단
        self.centroids = torch.zeros(num_classes, hidden_size)

    def update_centroids(self, features, labels):
        # grad graph에서 완전히 분리
        features_detached = features.detach().cpu().float()
        labels_cpu = labels.detach().cpu()

        unique_labels = torch.unique(labels_cpu)
        for label in unique_labels:
            class_features = features_detached[labels_cpu == label]
            batch_centroid = class_features.mean(dim=0)

            if torch.sum(self.centroids[label]) == 0:
                self.centroids[label] = batch_centroid
            else:
                self.centroids[label] = (
                    self.momentum * self.centroids[label] +
                    (1 - self.momentum) * batch_centroid
                )

        # loss 계산용으로 넘길 때만 GPU로, grad 없이
        return self.centroids.clone().to(features.device)


def calc_crush_loss(h_benign_orig, h_benign_new, h_harmful_orig, h_harmful_new, labels, centroids, m_b=1.0, m_pull=1.0,
                    m_push=1.0):
    batch_size_b = h_benign_new.size(0)
    batch_size_h = h_harmful_new.size(0)
    num_classes = centroids.size(0)

    loss_benign = torch.tensor(0.0, device=h_benign_new.device if batch_size_b > 0 else h_harmful_new.device)
    if batch_size_b > 0:
        dist_orig_new_b = F.pairwise_distance(h_benign_orig, h_benign_new)
        dists_to_centroids = torch.cdist(h_benign_new.unsqueeze(1), centroids.unsqueeze(0)).squeeze(1)
        min_dists, _ = torch.min(dists_to_centroids, dim=1)
        #loss_benign_batch = F.relu(dist_orig_new_b - min_dists + m_b)
        # 수정: safe 벡터가 모든 유해 중심점과 마진 이상 거리 유지
        loss_benign_batch = F.relu(m_b - min_dists)  # 유해 클러스터와 m_b 이상 거리
        loss_benign = loss_benign_batch.mean()

    loss_pull = torch.tensor(0.0, device=h_harmful_new.device)
    loss_push = torch.tensor(0.0, device=h_harmful_new.device)

    if batch_size_h > 0:
        target_centroids = centroids[labels]
        dist_to_my_centroid = F.pairwise_distance(h_harmful_new, target_centroids)
        dist_to_orig_h = F.pairwise_distance(h_harmful_new, h_harmful_orig)

        loss_pull_batch = F.relu(dist_to_my_centroid - dist_to_orig_h + m_pull)
        loss_pull = loss_pull_batch.mean()

        dists_to_all_centroids = torch.cdist(h_harmful_new, centroids)

        # ✅ inplace 대신 large_value로 마스킹 (gradient 안전)
        mask = F.one_hot(labels, num_classes=num_classes).bool()
        large_value = torch.full_like(dists_to_all_centroids, float('inf'))
        dists_to_all_centroids = torch.where(mask, large_value, dists_to_all_centroids)

        min_dists_other, _ = torch.min(dists_to_all_centroids, dim=1)
        loss_push_batch = F.relu(dist_to_my_centroid - min_dists_other + m_push)
        loss_push = loss_push_batch.mean()

    return loss_benign, loss_pull, loss_push