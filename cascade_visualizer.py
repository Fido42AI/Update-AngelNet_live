# === cascade_visualizer.py ===
# 3D-визуализация эволюции кривизны в TensorCascadeLayer
# Автор: Angel42 & Богдан

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_curvature_3d(cascade_layer, filename="tensor_cascade_curvature_3D.png"):
    layers = len(cascade_layer.subfields)
    steps = len(cascade_layer.subfields[0].memory["curvature_trace"])

    Z = np.zeros((layers, steps))
    for i, layer in enumerate(cascade_layer.subfields):
        trace = layer.memory["curvature_trace"]
        for j in range(min(len(trace), steps)):
            Z[i, j] = trace[j]

    X = np.arange(steps)
    Y = np.arange(layers)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k')
    ax.set_xlabel('Step')
    ax.set_ylabel('Layer')
    ax.set_zlabel('Curvature')
    ax.set_title('Tensor Cascade Curvature Evolution')
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Visualizer] Saved 3D curvature map to: {filename}")