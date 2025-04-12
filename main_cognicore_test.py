# === main_cognicore_test.py ===
# Тест AngelNet с CogniCore и визуализацией каскада

import torch
from angelnet_fiber_core import AngelNetFiberCore
from cascade_visualizer import plot_curvature_3d
from cognicore import CogniCore

def main():
    print("=== [AngelNet CogniCore Test] ===\n")

    net = AngelNetFiberCore(
        depth_levels=3,
        insert_tensor_field_at=1,
        base_curvature=-0.2,
        use_tensor_field_v3=True,
        use_feedback=True,
        use_tensor_cascade=True
    )

    cogni = CogniCore()

    inputs = [
        torch.tensor([[0.1000, 0.1000, 0.0500]]),
        torch.tensor([[1.5000, -1.0000, 0.5000]]),
        torch.tensor([[0.0500, 0.0500, 0.0500]]),
        torch.tensor([[2.0000, 0.0000, -2.0000]])
    ]

    for i, input_tensor in enumerate(inputs):
        print(f"--- [STEP {i+1}] ---")
        print(f"[Input]: {input_tensor}")

        net.clear_memory()
        net.add_input_tensor(input_tensor)

        # CogniCore анализ
        state, action = cogni.analyze(input_tensor)
        print(f"[CogniCore] State: {state} | Action: {action}")

        # Инъекция в каскад
        for layer in net.layers:
            if hasattr(layer, "inject_cognitive_adjustment"):
                layer.inject_cognitive_adjustment(action["adjustment"])

        output = net.forward()
        print(f"[Output]: {output}\n")

    # Сводка
    print("[Cascade Summary]")
    net.layers[1].summary()
    print("\n[CogniCore State Summary]")
    cogni.summary()

    # Визуализация кривизны
    plot_curvature_3d(net.layers[1])

if __name__ == "__main__":
    main()