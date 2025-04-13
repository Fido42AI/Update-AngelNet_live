import torch
from angelnet_fiber_core import AngelNetFiberCore


def select_mode():
    print("=== AngelNet Configurator ===")
    print("Select operating mode:")
    print("1 – Explorer")
    print("2 – Observer")
    print("3 – Anxiety Scanner")
    print("4 – Focus Fighter")
    print("5 – Manual Parameter Input")
    choice = input("Your choice (1–5): ")

    if choice == '1':
        return dict(base_curvature=-0.3, reaction_weight=0.2, target_mood=0.6)
    elif choice == '2':
        return dict(base_curvature=0.0, reaction_weight=0.1, target_mood=0.0)
    elif choice == '3':
        return dict(base_curvature=-0.5, reaction_weight=0.4, target_mood=-0.2)
    elif choice == '4':
        return dict(base_curvature=0.2, reaction_weight=0.3, target_mood=0.8)
    elif choice == '5':
        print("Enter parameters manually:")
        base_curvature = float(input("base_curvature (-1.0 → +1.0): "))
        reaction_weight = float(input("reaction_weight (0.0 → 1.0): "))
        target_mood = float(input("target_mood (-1.0 → +1.0): "))
        return dict(base_curvature=base_curvature, reaction_weight=reaction_weight, target_mood=target_mood)
    else:
        print("Invalid selection. Using default mode.")
        return dict(base_curvature=-0.2, reaction_weight=0.25, target_mood=0.5)


def main():
    config = select_mode()

    print("\n[Selected Parameters]")
    for k, v in config.items():
        print(f"{k}: {v}")

    net = AngelNetFiberCore(
        input_dim=3,
        hidden_dim=3,
        num_layers=3,
        use_tensor_field_v3=True,
        use_cascade=True,
        use_cognicore=True,
        memory_inertia=0.7,
        base_curvature=config["base_curvature"],
        num_fields=3,
        reaction_weight=config["reaction_weight"],
        target_mood=config["target_mood"]
    )

    input_tensor = torch.tensor([[0.1, 0.1, 0.05]])
    print("\n[STEP] Input tensor:", input_tensor)

    net.add_input_tensor(input_tensor, tags=["input"])
    output = net.forward()

    print("\n[Output]:", output)
    net.summary()

    if net.cogni:
        net.cogni.plot_mood_trace()
        net.cogni.plot_modumap()


if __name__ == "__main__":
    main()