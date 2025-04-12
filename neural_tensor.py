# === neural_tensor.py ===
# Neural Tensor v1.5 (метрика интегрирована)
# Автор: Angel42 & Богдан Федеренко

import torch
from metric_tensor import MetricTensor


class NeuralTensor:
    def __init__(self, data: torch.Tensor, depth: int = 0, scale: float = 1.0, tags=None, curvature: float = 0.0):
        self.data = data
        self.depth = depth
        self.scale = scale
        self.tags = tags or []
        self.metric = MetricTensor(data.shape[1], curvature)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str):
        if tag in self.tags:
            self.tags.remove(tag)

    def clone_scaled(self, scale_factor: float):
        return NeuralTensor(
            data=self.data * scale_factor,
            depth=self.depth,
            scale=self.scale * scale_factor,
            tags=self.tags.copy(),
            curvature=self.metric.curvature
        )

    def metric_norm(self):
        return self.metric.norm(self.data)

    def metric_cosine_with(self, other_tensor):
        return self.metric.cosine_similarity(self.data, other_tensor.data)

    def summary(self):
        print(f"\n[NeuralTensor] Depth={self.depth} | Scale={self.scale}")
        print(f"Tags: {self.tags}")
        print(f"Metric Norm: {self.metric_norm().item():.4f}")
        self.metric.summary()

    def __repr__(self):
        return f"<NeuralTensor depth={self.depth} scale={self.scale} shape={self.data.shape}>"