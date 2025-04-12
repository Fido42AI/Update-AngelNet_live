# === neural_container.py ===
# NeuralContainer: оболочка для работы с тензорами в контексте фрактальной сети
# Автор: Fido42AI & Богдан Федеренко

from neural_tensor import NeuralTensor

class NeuralContainer:
    def __init__(self):
        self.tensors = []

    def add_tensor(self, tensor: NeuralTensor):
        self.tensors.append(tensor)

    def filter_by_tag(self, tag: str):
        """
        Фильтрация тензоров по наличию конкретного тега
        """
        return [t for t in self.tensors if t.has_tag(tag)]

    def average(self):
        """
        Вычисление среднего значения всех тензоров
        """
        if not self.tensors:
            return None
        total = sum(t.data for t in self.tensors)
        return total / len(self.tensors)

    def propagate_scale(self):
        """
        Применение масштабирующего коэффициента ко всем тензорам
        """
        for tensor in self.tensors:
            tensor.data = tensor.data * tensor.scale

    def clear(self):
        """
        Очистка всех тензоров из контейнера
        """
        self.tensors.clear()

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __repr__(self):
        return f"<NeuralContainer with {len(self.tensors)} tensors>"