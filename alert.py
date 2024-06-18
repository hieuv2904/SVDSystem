import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.hidden = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# model = SimpleANN()

# model.load_state_dict(torch.load(r'D:\yowov2V7\YOWOv2\model_weights.pth'))

# model.eval()
# outputs = model(X_batch)  # đưa cái chuỗi 16 cái có hay ko vô đây 
# predicted = outputs.round()