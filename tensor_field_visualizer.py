# === tensor_field_visualizer.py ===
# TensorFieldVisualizer: визуализация фрактального поля
# Автор: Angel42 & Богдан Федеренко

import torch
import matplotlib.pyplot as plt
import numpy as np
from tensor_field_layer import TensorFieldLayer


class TensorFieldVisualizer:
    def __init__(self, field_layer: TensorFieldLayer, resolution=40):
        self.layer = field_layer
        self.resolution = resolution  # количество точек по оси

    def generate_grid(self):
        r = 1.5  # диапазон [-r, r]
        x = np.linspace(-r, r, self.resolution)
        y = np.linspace(-r, r, self.resolution)
        grid_x, grid_y = np.meshgrid(x, y)
        return grid_x, grid_y

    def compute_field(self):
        gx, gy = self.generate_grid()
        U = np.zeros_like(gx)
        V = np.zeros_like(gy)

        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                inp = torch.tensor([[gx[i, j], gy[i, j], 0.0]], dtype=torch.float32)
                out = self.layer(inp).detach().numpy()[0]
                U[i, j] = out[0]
                V[i, j] = out[1]
        return gx, gy, U, V

    def plot(self, title="Tensor Field Visualization"):
        gx, gy, U, V = self.compute_field()

        plt.figure(figsize=(8, 8))
        plt.quiver(gx, gy, U, V, color="darkblue", angles='xy')
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.axis('equal')
        plt.show()