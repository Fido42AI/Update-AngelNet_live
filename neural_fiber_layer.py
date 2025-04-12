import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_container import NeuralContainer


class NeuralFiberLayer(nn.Module):
    def __init__(self, input_size=3, output_size=3, depth=3):
        super(NeuralFiberLayer, self).__init__()
        self.depth = depth
        self.input_size = input_size
        self.output_size = output_size

        self.sublayers = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(depth)
        ])

        self.axon_activation = nn.Tanh()
        self.dendrite_activation = nn.Sigmoid()
        self.training_examples = []
        self.covectors = []

        self.input_container = NeuralContainer()
        self.output_container = NeuralContainer()

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Превращаем (3,) → (1, 3)
        print(f"[NeuralFiberLayer] Input shape: {x.shape}")
        out = x
        for i, layer in enumerate(self.sublayers):
            print(f"[NeuralFiberLayer] Sublayer {i} input shape: {out.shape}")
            axon = self.axon_activation(layer(out))
            dendrite = self.dendrite_activation(layer(out))
            out = axon * dendrite
            print(f"[NeuralFiberLayer] Sublayer {i} output shape: {out.shape}")
        return out

    def process(self):
        self.output_container.tensors.clear()
        for tensor in self.input_container.tensors:
            result = self.forward(tensor.data)
            self.output_container.add_tensor(
                tensor.clone_scaled(1.0)
            )
            self.output_container.tensors[-1].data = result

    def add_example(self, input_vector, expected_output):
        self.training_examples.append((input_vector, expected_output))

    def train_custom(self, epochs=1, learning_rate=0.01):
        for epoch in range(epochs):
            for x, y in self.training_examples:
                pass  # Placeholder for custom training logic

    def add_covector(self, covector, label=""):
        self.covectors.append((label, covector))
        print(f"[NeuralFiberLayer] CoVector '{label}' registered.")

    def clear_io(self):
        self.input_container.tensors.clear()
        self.output_container.tensors.clear()