import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b3
import sys
sys.path.append(".")
from utils.hierachy_config import DEVICE
from model_code.fen_model import fen_model




class Buffer_switcher_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, outsider_hidden_size, action_num):
        super(Buffer_switcher_model, self).__init__()
        self.fen_model = fen_model(hidden_size1, hidden_size1)
        self.outsider_fen_model = efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier = nn.Linear(1280, outsider_hidden_size)
        self.outsider_contrast_fc = nn.Linear(2 * outsider_hidden_size, outsider_hidden_size)
        self.outsider_fc = nn.Linear(9 * outsider_hidden_size, outsider_hidden_size)
        self.fc1 = nn.Linear(hidden_size1 + outsider_hidden_size, hidden_size2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn = nn.BatchNorm1d(hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, action_num)

    def forward(self, image, outsider_piece, mask=None):
        batch_size = image.size(0)
        image_input = self.fen_model(image, mask)
        outsider_input = self.outsider_fen_model(outsider_piece)

        outsider_image_tensor = image.unfold(2, 96, 96).unfold(3, 96, 96)
        outsider_image_tensor = outsider_image_tensor.permute(0, 2, 3, 1, 4, 5).contiguous()
        outsider_image_tensor = outsider_image_tensor.view(batch_size * 9, -1, 96, 96)
        outsider_image_tensor = self.outsider_fen_model(outsider_image_tensor)
        outsider_image_tensor = outsider_image_tensor.view(batch_size, 9, -1)
        outsider_input = outsider_input.unsqueeze(1).expand(batch_size, 9, -1)

        outsider_tensor = self.outsider_contrast_fc(torch.cat([outsider_input, outsider_image_tensor], dim=-1))
        if mask is not None:
            mask_matrix = torch.ones(batch_size, 9, device=DEVICE)
            for batch_id in range(len(mask)):
                for mask_value in mask[batch_id]:
                    if mask_value == 9:
                        mask_matrix[batch_id] = 0
                    else:
                        mask_matrix[batch_id][mask_value] = 0
            outsider_tensor = outsider_tensor * mask_matrix.unsqueeze(-1)

        outsider_tensor = outsider_tensor.view(batch_size, -1)
        outsider_tensor = self.outsider_fc(outsider_tensor)
        feature_tensor = torch.cat([image_input, outsider_tensor], dim=-1)
        out = self.fc1(feature_tensor)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Decider_model(nn.Module):
    def __init__(self, fen_model_hidden1, fen_model_hidden2, outsider_hidden, hidden_1, hidden_2, action_num, dropout=0.1):
        super().__init__()
        self.fen_model = fen_model(fen_model_hidden1, fen_model_hidden2)
        self.outsider_fen = efficientnet_b0(weights="DEFAULT")
        self.outsider_fen.classifier = nn.Linear(1280, outsider_hidden)
        self.outsider_contrast_fc = nn.Linear(2 * outsider_hidden, outsider_hidden)
        self.outsider_fc = nn.Linear(9 * outsider_hidden, outsider_hidden)
        self.fc1 = nn.Linear(fen_model_hidden2 + outsider_hidden, hidden_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.outlayer = nn.Linear(hidden_2, action_num)

    def forward(self, image, outsider_piece, mask=None):
        batch_size = image.size(0)
        image_input = self.fen_model(image, mask)
        outsider_input = self.outsider_fen(outsider_piece)

        outsider_image_tensor = image.unfold(2, 96, 96).unfold(3, 96, 96)
        outsider_image_tensor = outsider_image_tensor.permute(0, 2, 3, 1, 4, 5).contiguous()
        outsider_image_tensor = outsider_image_tensor.view(batch_size * 9, -1, 96, 96)
        outsider_image_tensor = self.outsider_fen(outsider_image_tensor)
        outsider_image_tensor = outsider_image_tensor.view(batch_size, 9, -1)
        outsider_input = outsider_input.unsqueeze(1).expand(batch_size, 9, -1)

        outsider_tensor = self.outsider_contrast_fc(torch.cat([outsider_input, outsider_image_tensor], dim=-1))
        if mask is not None:
            mask_matrix = torch.ones(batch_size, 9, device=DEVICE)
            for batch_id in range(len(mask)):
                for mask_value in mask[batch_id]:
                    if mask_value == 9:
                        mask_matrix[batch_id] = 0
                    else:
                        mask_matrix[batch_id][mask_value] = 0
            outsider_tensor = outsider_tensor * mask_matrix.unsqueeze(-1)

        outsider_tensor = outsider_image_tensor.view(batch_size, -1)
        outsider_tensor = self.outsider_fc(outsider_tensor)
        feature_tensor = torch.cat([image_input, outsider_tensor], dim=-1)
        out = self.fc1(feature_tensor)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.outlayer(out)
        return out


class Local_switcher_model(nn.Module):
    def __init__(self, fen_model_hidden1, fen_model_hidden2, hidden1, hidden2, action_num, dropout=0.1):
        super().__init__()
        self.fen_model = fen_model(fen_model_hidden1, fen_model_hidden2)
        self.fc1 = nn.Linear(fen_model_hidden2, hidden1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.do = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.outlayer = nn.Linear(hidden2, action_num)

    def forward(self, image, mask=None):
        feature_tensor = self.fen_model(image, mask)
        out = self.fc1(feature_tensor)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.do(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.outlayer(out)
        return out
