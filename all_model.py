import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch.nn.functional as F

class GraphModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, model_type='GAT', hidden=16, heads=4):
        super(GraphModel, self).__init__()
        self.model_type = model_type
        if model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden, heads=heads)
            self.conv2 = GATConv(hidden * heads, out_channels)
        elif model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, out_channels)
        elif model_type == 'GIN':
            self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)))
            self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden, out_channels), torch.nn.ReLU(), torch.nn.Linear(out_channels, out_channels)))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, data, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
        pred = out.argmax(dim=1)
        correct = int(pred[mask].eq(data.y[mask]).sum().item())
        acc = correct / int(mask.sum())
        return acc



# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集列表
datasets = ['Cora', 'Citeseer']

# 定义模型、优化器和损失函数
model_type = 'GAT'  # 可以选择 'GAT', 'GCN', 'GraphSAGE', 'GIN' 中的任意一个
hidden_channels = 16
heads = 4
num_epochs = 200
lr = 0.01
weight_decay = 5e-4

for name in datasets:
    print(f"Training on {name} dataset")

    # 加载数据集
    dataset = Planetoid(root=f'./data/{name}', name=name)
    data = dataset[0].to(device)



    # 实例化模型
    model = GraphModel(in_channels=dataset.num_node_features, out_channels=dataset.num_classes,
                       model_type=model_type, hidden=hidden_channels, heads=heads).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # 实例化Trainer类
    trainer = Trainer(model, optimizer, criterion, device)

    # 训练模型
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(num_epochs):
        loss = trainer.train(data)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

        # 在每个epoch结束时评估模型
        val_acc = trainer.evaluate(data, data.val_mask)
        print(f'Validation Accuracy: {val_acc:.4f}')

        # 保存最佳验证集准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = trainer.evaluate(data, data.test_mask)
            print(f'Best Test Accuracy: {best_test_acc:.4f}')

    # 输出最佳测试集准确率
    print(f"{name} dataset - Best Test Accuracy: {best_test_acc:.4f}\n")