
# === cognicore_interpreter.py ===
# Модуль интерпретации кластеров и кривизны для CogniCore
# Автор: Angel42 & Богдан

class CogniCoreInterpreter:
    def __init__(self):
        self.last_state = None

    def interpret(self, cluster_data, curvature_data):
        """
        На вход: словарь кластеров и история кривизны
        На выход: когнитивное состояние и реакция
        """
        le = cluster_data.get("LE", 0)
        hf = cluster_data.get("HF", 0)
        oc = cluster_data.get("OC", 0)

        avg_curvature = sum(curvature_data) / len(curvature_data) if curvature_data else 0

        # Анализ и порождение состояния
        if le > 2 and avg_curvature < -0.22:
            state = "depressed"
            action = {"adjustment": +0.05, "comment": "Усталость — поле гаснет"}
        elif hf > 1:
            state = "stressed"
            action = {"adjustment": -0.05, "comment": "Перегруз — поле напрягается"}
        elif oc > 1:
            state = "anxious"
            action = {"adjustment": -0.03, "comment": "Колебания — нестабильность"}
        else:
            state = "neutral"
            action = {"adjustment": 0.0, "comment": "Стабильность"}

        self.last_state = state
        return state, action

    def summary(self):
        print(f"[CogniCoreInterpreter] Last state: {self.last_state}")