import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Menetapkan perangkat ke CPU
device = torch.device('cpu')

# Membuat model dan memindahkannya ke CPU
model = NeuralNet(input_size=10, hidden_size=20, num_classes=3).to(device)

# Membuat tensor dan memindahkannya ke CPU
tensor = torch.randn(3, 3).to(device)

print(model)
print(tensor)
