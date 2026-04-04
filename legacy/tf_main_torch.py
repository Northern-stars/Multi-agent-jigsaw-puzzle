import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import models
from tqdm import tqdm
from utils import plot_confusion_matrix


# ========= 数据准备 ==========
train_x_path = 'dataset/train_img_48gap_33-001.npy'
train_y_path = 'dataset/train_label_48gap_33.npy'
train_data_x = np.load(train_x_path)
train_data_y = np.load(train_y_path)
test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'
test_data_x = np.load(test_x_path)
test_data_y = np.load(test_y_path)

train_x_1 = np.concatenate((train_data_x[:, :96, :96, :], train_data_x[:, :96, 96:192, :],
                            train_data_x[:, :96, 192:, :], train_data_x[:, 96:192, :96, :],
                            train_data_x[:, 96:192, 192:, :], train_data_x[:, 192:, :96, :],
                            train_data_x[:, 192:, 96:192, :], train_data_x[:, 192:, 192:, :]))# 8 pieces 
train_x_2 = np.tile(train_data_x[:, 96:192, 96:192, :], (8, 1, 1, 1))# copy the central piece 8 times
train_y = np.concatenate([train_data_y[:, i, :] for i in range(8)])# use the one-hot category as label

# 转成 PyTorch tensor
train_x_1 = torch.tensor(train_x_1, dtype=torch.float32).permute(0,3,1,2) / 255.0
train_x_2 = torch.tensor(train_x_2, dtype=torch.float32).permute(0,3,1,2) / 255.0
train_y = torch.tensor(np.argmax(train_y, axis=1), dtype=torch.long)  # one-hot → class id

train_dataset = TensorDataset(train_x_1, train_x_2, train_y)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

test_x_1 = np.concatenate((test_data_x[:, :96, :96, :], test_data_x[:, :96, 96:192, :],
                           test_data_x[:, :96, 192:, :], test_data_x[:, 96:192, :96, :],
                           test_data_x[:, 96:192, 192:, :], test_data_x[:, 192:, :96, :],
                           test_data_x[:, 192:, 96:192, :], test_data_x[:, 192:, 192:, :]))
test_x_2 = np.tile(test_data_x[:, 96:192, 96:192, :], (8, 1, 1, 1))
test_y = np.concatenate([test_data_y[:, i, :] for i in range(8)])
x1 = torch.tensor(test_x_1, dtype=torch.float32).permute(0,3,1,2) / 255.0
x2 = torch.tensor(test_x_2, dtype=torch.float32).permute(0,3,1,2) / 255.0
y = torch.tensor(np.argmax(test_y, axis=1), dtype=torch.long)

test_dataset=TensorDataset(x1,x2,y)
test_dataloader= DataLoader(test_dataset, batch_size=100, shuffle=True)

FILE_NAME="_tf_main"
from os import path
MODEL_PATH=path.join("model","tf_combonet.pth")
LOAD=False


# ========= 模型定义 ==========
class ComboNet(nn.Module):
    def __init__(self, feature_dim=512, num_classes=8):
        super(ComboNet, self).__init__()
        # EfficientNetB0 backbone
        base_model = models.efficientnet_b0(weights=None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # 去掉分类头
        in_features = base_model.classifier[1].in_features

        self.fc_feature = nn.Linear(in_features, feature_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f1 = f1.flatten(1)
        f1 = self.fc_feature(f1)

        f2 = self.feature_extractor(x2)
        f2 = f2.flatten(1)
        f2 = self.fc_feature(f2)

        feat = torch.cat([f1, f2], dim=1)
        out = self.classifier(feat)
        return out


# ========= 训练 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComboNet().to(DEVICE)


def train(model:nn.Module,load,epoch_num=20):
    if load:
        model.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epoch_num):
        model.train()
        total_loss, total_correct = 0, 0
        for x1, x2, y in tqdm(train_loader):
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()

        acc = total_correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader.dataset):.4f}, Acc={acc:.4f}")
        if epoch % 5 == 0:
            test(model)
            torch.save(model.state_dict(),MODEL_PATH)

def test(model:nn.Module,plot=False):
    model.eval()
    confusion_matrix=torch.zeros([8,8])
    with torch.no_grad():
        total_correct = 0
        for x1, x2, y in tqdm(test_dataloader):
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            logits = model(x1, x2)
            pred_label=logits.argmax(1)
            total_correct += (pred_label == y).sum().item()
            for pred,label in zip(pred_label.cpu(),y.cpu()):
                confusion_matrix[label,pred]+=1
        acc = total_correct / len(test_dataloader.dataset)
        print(f"Test Acc: {acc:.4f}")
    if plot:
        plot_confusion_matrix(confusion_matrix,list(range(8)),FILE_NAME)


if __name__=="__main__":
    train(model,LOAD,20)
    test(model,True)