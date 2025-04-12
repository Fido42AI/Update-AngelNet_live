# === metric_tensor.py ===
# Модуль метрического тензора gᵢⱼ для AngelNet v1.5
# Автор: Angel42 & Богдан Федеренко

import torch

class MetricTensor:
    def __init__(self, size: int, curvature: float = 0.0):
        self.size = size
        self.curvature = curvature
        self.tensor = self._build_metric_tensor()

    def _build_metric_tensor(self):
        base = torch.eye(self.size)
        if self.curvature > 0:
            scale = 1.0 + self.curvature
        elif self.curvature < 0:
            scale = 1.0 / (1.0 - self.curvature)
        else:
            scale = 1.0
        return base * scale

    def apply(self, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        g = self.tensor.to(v.device)
        return torch.matmul(torch.matmul(v, g), w.T)

    def apply_to_vector(self, v: torch.Tensor) -> torch.Tensor:
        """
        Новый метод: применяет метрический тензор к вектору,
        возвращает деформированный вектор: gᵢⱼ * vⱼ
        """
        g = self.tensor.to(v.device)
        return torch.matmul(v, g)

    def norm(self, v: torch.Tensor) -> torch.Tensor:
        g = self.tensor.to(v.device)
        return torch.matmul(torch.matmul(v, g), v.T).sqrt()

    def cosine_similarity(self, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        num = self.apply(v, w)
        denom = self.norm(v) * self.norm(w)
        return num / (denom + 1e-8)

    def summary(self):
        print(f"[MetricTensor] Size: {self.size}, Curvature: {self.curvature}")
        print(f"Tensor gᵢⱼ:\n{self.tensor}")