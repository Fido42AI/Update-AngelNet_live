# === cognicore_state.py ===
# Модуль долгосрочного состояния CogniCore
# Автор: Angel42 & Богдан

class CogniCoreState:
    def __init__(self):
        self.mood = "neutral"
        self.mood_score = 0.0
        self.history = []

    def update_state(self, new_state):
        """
        Обновление внутреннего состояния в зависимости от нового когнитивного состояния
        """
        weight_map = {
            "depressed": -0.3,
            "stressed": -0.2,
            "anxious": -0.1,
            "neutral": +0.05,
            "happy": +0.3,
            "focused": +0.2
        }

        delta = weight_map.get(new_state, 0.0)
        self.mood_score += delta
        self.mood_score = max(-1.0, min(1.0, self.mood_score))  # ограничиваем в пределах [-1, +1]

        if self.mood_score < -0.5:
            self.mood = "negative"
        elif self.mood_score > +0.5:
            self.mood = "positive"
        else:
            self.mood = "neutral"

        self.history.append((new_state, self.mood_score))

    def summary(self):
        print(f"[CogniCoreState] Mood: {self.mood} | Score: {self.mood_score:.2f}")
        print("Recent states:")
        for entry in self.history[-5:]:
            print("  -", entry)