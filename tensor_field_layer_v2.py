import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from metric_tensor import MetricTensor


class TensorFieldLayer(nn.Module):
    def __init__(self, in_features, out_features, base_curvature=-0.2, phase=0.5, rotation=0.25,
                 adapt_rate=0.05, feedback_weight=0.3, memory_inertia=0.7, max_memory=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_curvature = base_curvature
        self.curvature = base_curvature
        self.phase = phase
        self.rotation = rotation
        self.adapt_rate = adapt_rate
        self.feedback_weight = feedback_weight
        self.memory_inertia = memory_inertia
        self.baseline_energy = 0.5 * in_features

        self.metric = MetricTensor(size=in_features, curvature=self.curvature)
        self.linear = nn.Linear(in_features, out_features)

        self.memory = {
            "curvature_trace": [],
            "energy_trace": [],
            "feedback_trace": [],
            "clusters": {
                "low_energy": [],
                "high_feedback": [],
                "oscillating_curvature": []
            },
            "max_memory": max_memory
        }

    def _rotate_input(self, x):
        angle = torch.tensor(self.phase * math.pi + self.rotation, dtype=torch.float32, device=x.device)
        rotation_matrix = torch.eye(self.in_features, device=x.device)
        cos_val, sin_val = torch.cos(angle), torch.sin(angle)
        if self.in_features >= 2:
            rotation_matrix[0, 0] = cos_val
            rotation_matrix[0, 1] = -sin_val
            rotation_matrix[1, 0] = sin_val
            rotation_matrix[1, 1] = cos_val
        return x @ rotation_matrix

    def forward(self, x, external_feedback=None):
        energy = torch.sum(torch.abs(x)).item()
        feedback_energy = torch.sum(torch.abs(external_feedback)).item() if external_feedback is not None else 0.0

        self._update_memory(energy, feedback_energy)
        self._update_clusters()

        delta_curvature = self.adapt_rate * (energy - self.baseline_energy)
        delta_curvature += self.feedback_weight * self.adapt_rate * (feedback_energy - self.baseline_energy)

        cluster_modifier = self._cluster_modulation(energy, feedback_energy)
        delta_curvature *= cluster_modifier

        new_curvature = self.base_curvature + delta_curvature
        self.curvature = self.memory_inertia * self.curvature + (1.0 - self.memory_inertia) * new_curvature
        self.metric = MetricTensor(size=self.in_features, curvature=self.curvature)

        rotated = self._rotate_input(x)
        projected = self.linear(rotated)
        deformed = self.metric.apply_to_vector(projected)
        return deformed

    def _update_memory(self, energy, feedback_energy):
        mem = self.memory
        mem["curvature_trace"].append(self.curvature)
        mem["energy_trace"].append(energy)
        mem["feedback_trace"].append(feedback_energy)
        for key in ["curvature_trace", "energy_trace", "feedback_trace"]:
            if len(mem[key]) > mem["max_memory"]:
                mem[key] = mem[key][-mem["max_memory"]:]

    def _update_clusters(self):
        mem = self.memory
        if len(mem["energy_trace"]) < 2:
            return

        energy = mem["energy_trace"][-1]
        feedback = mem["feedback_trace"][-1]
        curvature_change = abs(mem["curvature_trace"][-1] - mem["curvature_trace"][-2])

        if energy < self.baseline_energy * 0.8:
            mem["clusters"]["low_energy"].append(energy)
        if feedback > energy and feedback > self.baseline_energy:
            mem["clusters"]["high_feedback"].append(feedback)
        if curvature_change > 0.05:
            mem["clusters"]["oscillating_curvature"].append(curvature_change)

        for key in mem["clusters"]:
            if len(mem["clusters"][key]) > mem["max_memory"]:
                mem["clusters"][key] = mem["clusters"][key][-mem["max_memory"]:]

    def _cluster_modulation(self, energy, feedback_energy):
        mem = self.memory["clusters"]
        factor = 1.0
        if any(abs(e - energy) < 0.1 for e in mem["low_energy"]):
            factor *= 0.8
        if any(abs(f - feedback_energy) < 0.1 for f in mem["high_feedback"]):
            factor *= 1.3
        if len(mem["oscillating_curvature"]) >= 3:
            factor *= 1.1
        return factor

    def plot_curvature_trace(self):
        trace = self.memory["curvature_trace"]
        if not trace:
            print("[TensorFieldLayer] Нет данных кривизны.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(trace, label='Curvature', linewidth=2)
        plt.axhline(self.base_curvature, color='gray', linestyle='--', label='Base')
        plt.title("TensorFieldLayer — Кривая кривизны")
        plt.xlabel("Шаги")
        plt.ylabel("Кривизна")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_cluster_map(self):
        mem = self.memory["clusters"]
        labels = ['Low Energy', 'High Feedback', 'Oscillation']
        sizes = [len(mem["low_energy"]), len(mem["high_feedback"]), len(mem["oscillating_curvature"])]
        if sum(sizes) == 0:
            print("[TensorFieldLayer] Кластеры пусты.")
            return

        colors = ['#77d977', '#ff7777', '#aaaaff']
        plt.figure(figsize=(5, 5))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title("Карта кластеров TensorFieldMemory")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return (f"<TensorFieldLayer: {self.in_features} → {self.out_features} | "
                f"curvature: base={self.base_curvature:+.2f}, now={self.curvature:+.2f}, "
                f"feedback_weight={self.feedback_weight:.2f}, inertia={self.memory_inertia:.2f}, "
                f"clusters={{LE:{len(self.memory['clusters']['low_energy'])}, "
                f"HF:{len(self.memory['clusters']['high_feedback'])}, "
                f"OC:{len(self.memory['clusters']['oscillating_curvature'])}}}>")