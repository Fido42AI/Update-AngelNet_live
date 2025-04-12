# === tensor_cascade_layer.py ===
# TensorCascadeLayer: Каскад из нескольких TensorFieldLayer
# Автор: Angel42 & Богдан

import torch.nn as nn
from tensor_field_layer_v3 import TensorFieldLayer

class TensorCascadeLayer(nn.Module):
    def __init__(self, in_features, out_features, base_curvature=-0.2, num_layers=3,
                 feedback_weight=0.3, memory_inertia=0.7):
        super().__init__()
        self.subfields = nn.ModuleList([
            TensorFieldLayer(in_features, out_features,
                             base_curvature=base_curvature,
                             feedback_weight=feedback_weight,
                             memory_inertia=memory_inertia)
            for _ in range(num_layers)
        ])
        self.last_cognitive_adjustment = 0.0

    def forward(self, x, cognitive_adjustment=0.0):
        self.last_cognitive_adjustment = cognitive_adjustment
        out = x
        for layer in self.subfields:
            out = layer(out, external_feedback=out)
        return out

    def inject_cognitive_adjustment(self, value: float):
        self.last_cognitive_adjustment = value

    def summary(self):
        print("=== TensorCascadeLayer Summary ===")
        for i, layer in enumerate(self.subfields):
            print(f" SubLayer {i}: {layer}")

    def aggregate_clusters(self):
        result = {"LE": 0, "HF": 0, "OC": 0}
        for layer in self.subfields:
            mem = layer.memory["clusters"]
            result["LE"] += len(mem["low_energy"])
            result["HF"] += len(mem["high_feedback"])
            result["OC"] += len(mem["oscillating_curvature"])
        return result