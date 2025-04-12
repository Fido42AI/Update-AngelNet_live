# === angelnet_fiber_core.py ===
# Обновлённый модуль AngelNetFiberCore с поддержкой TensorCascade и TensorField v3
# Автор: Angel42 & Богдан

from neural_fiber_layer import NeuralFiberLayer
from neural_tensor import NeuralTensor
from neural_container import NeuralContainer
from tensor_field_layer_v3 import TensorFieldLayer
from tensor_cascade_layer import TensorCascadeLayer

class AngelNetFiberCore:
    def __init__(self, depth_levels=3, insert_tensor_field_at=1, base_curvature=-0.2,
                 use_tensor_field_v3=True, use_feedback=True, use_tensor_cascade=False):
        self.layers = []
        self.input_container = NeuralContainer()
        self.output_container = NeuralContainer()
        self.last_cognitive_adjustment = 0.0

        for i in range(depth_levels):
            if use_tensor_cascade and i == insert_tensor_field_at:
                layer = TensorCascadeLayer(
                    in_features=3,
                    out_features=3,
                    base_curvature=base_curvature,
                    feedback_weight=0.3,
                    memory_inertia=0.7
                )
            elif use_tensor_field_v3 and i == insert_tensor_field_at:
                layer = TensorFieldLayer(
                    in_features=3,
                    out_features=3,
                    base_curvature=base_curvature,
                    phase=0.5,
                    rotation=0.25,
                    adapt_rate=0.05,
                    feedback_weight=0.3,
                    memory_inertia=0.7
                )
            else:
                layer = NeuralFiberLayer(depth=i)
            self.layers.append(layer)

    def add_input_tensor(self, data, tags=None):
        tensor = NeuralTensor(data=data, tags=tags)
        self.input_container.add_tensor(tensor)
        return tensor

    def forward(self):
        current_tensor = self.input_container.tensors[0].data
        for layer in self.layers:
            if hasattr(layer, 'clear_io'):
                layer.clear_io()
            if hasattr(layer, "forward"):
                if "cognitive_adjustment" in layer.forward.__code__.co_varnames:
                    current_tensor = layer(current_tensor, cognitive_adjustment=self.last_cognitive_adjustment)
                else:
                    current_tensor = layer(current_tensor)
        self.output_container.tensors = [NeuralTensor(data=current_tensor)]
        return current_tensor

    def clear_memory(self):
        self.input_container.clear()
        self.output_container.clear()
        for layer in self.layers:
            if hasattr(layer, 'clear_io'):
                layer.clear_io()

    def summary(self):
        print("=== AngelNet Fiber Summary ===")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer}")

    def inject_cognitive_adjustment(self, value: float):
        self.last_cognitive_adjustment = value