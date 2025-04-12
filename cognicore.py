# === cognicore.py ===
# Модуль CogniCore — когнитивное ядро AngelNet
# Автор: Angel42 & Богдан

class CogniCoreState:
    def __init__(self):
        self.mood = "neutral"
        self.score = 0.0
        self.history = []

    def update(self, energy, curvature_shift):
        score = energy * 0.1 - curvature_shift * 0.2
        self.score += score
        self.history.append((self.mood, round(self.score, 4)))

        # Обновление настроения
        if self.score < -0.1:
            self.mood = "depressed"
        elif self.score > 0.2:
            self.mood = "curious"
        else:
            self.mood = "neutral"

        return self.mood

    def summary(self):
        print(f"[CogniCoreState] Mood: {self.mood} | Score: {round(self.score, 2)}")
        print("Recent states:")
        for h in self.history[-5:]:
            print(f"  - {h}")

class CogniCore:
    def __init__(self):
        self.state = CogniCoreState()

    def analyze(self, input_tensor):
        energy = input_tensor.abs().sum().item()
        curvature_shift = 0.05 if energy < 0.2 else (-0.05 if energy > 2.0 else 0.0)

        mood = self.state.update(energy, curvature_shift)

        if mood == "depressed":
            return mood, {"adjustment": 0.05, "comment": "Усталость — поле гаснет"}
        elif mood == "curious":
            return mood, {"adjustment": -0.03, "comment": "Интерес — усилить отклик"}
        else:
            return mood, {"adjustment": 0.0, "comment": "Стабильность"}

    def summary(self):
        self.state.summary()