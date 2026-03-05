import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from piece_compare import piece_compare_modulator as compare_model
# from deepuzzle_pretrain_modified import DeepuzzleModel as compare_model
import random
import numpy as np
import os
from utils import plot_confusion_matrix

LOAD=False
MODEL_PATH=os.path.join("model","deepuzzle9_modulator.pth")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TEST_PER_EPOCH=1
BATCH_SIZE=64
EPOCH_NUM=200
FILENAME="_deepuzzle_9_modulator"
LR=1e-4
ACTION_NUM=9

train_x_path = 'dataset/train_img_48gap_33-001.npy'
train_y_path = 'dataset/train_label_48gap_33.npy'
# test_x_path = 'dataset/train_img_48gap_33-001.npy'
# test_y_path = 'dataset/train_label_48gap_33.npy'
test_x_path = 'dataset/test_img_48gap_33.npy'
test_y_path = 'dataset/test_label_48gap_33.npy'
# test_x_path = 'dataset/valid_img_48gap_33.npy'
# test_y_path = 'dataset/valid_label_48gap_33.npy'

train_x=np.load(train_x_path)
train_y=np.load(train_y_path)

test_x=np.load(test_x_path)
test_y=np.load(test_y_path)

class DeepuzzleDataset(Dataset):
    def __init__(self,image,label):
        super().__init__()
        self.image=image
        self.label=label
        self.sector_num=self.image.shape[0]//3
    
    def __len__(self):
        return self.image.shape[0]
    
    def __getitem__(self, index):
        image1=torch.tensor(self.image[index]).permute([2,0,1]).to(torch.float)
        image1_label=list(self.label[index])
        for a in range(8):
            for b in range(8):
                # print(type(imageLabel[i]))
                # print(imageLabel)
                if image1_label[a][b]==True:
                    if b>=4:
                        image1_label[a]=b+1
                        break
                    else:
                        image1_label[a]=b
                        break
        image1_label.insert(4,4)
        random_index=random.randint(0,self.image.shape[0]-1)
        while random_index//self.sector_num==index//self.sector_num:
            random_index=random.randint(0,self.image.shape[0]-1)
        image2=torch.tensor(self.image[random_index]).permute([2,0,1]).to(torch.float)



        image1_fragments=[
            image1[:,0:96,0:96],
            image1[:,0:96,96:192],
            image1[:,0:96,192:288],
            image1[:,96:192,0:96],
            image1[:,96:192,96:192],
            image1[:,96:192,192:288],
            image1[:,192:288,0:96],
            image1[:,192:288,96:192],
            image1[:,192:288,192:288]
        ]
        image2_fragments=[
            image2[:,0:96,0:96],
            image2[:,0:96,96:192],
            image2[:,0:96,192:288],
            image2[:,96:192,0:96],
            image2[:,96:192,96:192],
            image2[:,96:192,192:288],
            image2[:,192:288,0:96],
            image2[:,192:288,96:192],
            image2[:,192:288,192:288]
        ]

        central_piece=image1_fragments[4]
        image2_fragments.pop(4)

        pos=random.randint(0,ACTION_NUM-1)
        if pos==8:
            piece=random.choice(image2_fragments)
        elif pos>=4:
            piece=image1_fragments[image1_label.index(pos+1)]
        else:
            piece=image1_fragments[image1_label.index(pos)]
        
        if torch.max(central_piece)>1:
            central_piece=central_piece/255
        if torch.max(piece)>1:
            piece=piece/255

        return central_piece,piece,pos
# class DeepuzzleDataset(Dataset):
#     def __init__(self, image, label):
#         # print(f"Loading data from {x_path}...")
#         self.image =image
#         self.label = label
#         self.num_samples = self.image.shape[0]
#         # 假设数据是按某种规律排列的，这里简单处理 sector
#         self.sector_num = max(1, self.num_samples // 3)
#         print("Data loaded.")

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, index):
#         # 1. 获取主图 (Host)
#         img1_np = self.image[index]

#         # 解析 Label (Permutation)
#         # label[index] 是 one-hot [8, 8]，转为 index list
#         # image1_label[i] 表示第 i 个位置放的是原来的第几号块
#         perm_onehot = self.label[index]
#         perm_idx = list(np.argmax(perm_onehot, axis=1))
#         # 修正 label (因为中间少了4号)
#         for i in range(len(perm_idx)):
#             if perm_idx[i] >= 4:
#                 perm_idx[i] += 1
#         perm_idx.insert(4, 4)  # 现在的 perm_idx 是 9 个数，对应 0-8 位置的块ID

#         # 2. 获取异图 (Guest) - 用于 Outsider
#         rand_idx = random.randint(0, self.num_samples - 1)
#         while rand_idx // self.sector_num == index // self.sector_num:
#             rand_idx = random.randint(0, self.num_samples - 1)
#         img2_np = self.image[rand_idx]

#         # 3. 切片 (Host)
#         # 注意：这里切出来的是视觉上的 9 个格子的内容
#         host_patches = []
#         for r in range(3):
#             for c in range(3):
#                 patch = (
#                     torch.tensor(
#                         img1_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]
#                     )
#                     .permute(2, 0, 1)
#                     .float()
#                 )
#                 host_patches.append(patch)

#         # 4. 切片 (Guest)
#         guest_patches = []
#         for r in range(3):
#             for c in range(3):
#                 patch = (
#                     torch.tensor(
#                         img2_np[r * 96 : (r + 1) * 96, c * 96 : (c + 1) * 96, :]
#                     )
#                     .permute(2, 0, 1)
#                     .float()
#                 )
#                 guest_patches.append(patch)

#         # 5. 确定 Center
#         center_piece = host_patches[4]

#         # 6. 构造样本
#         # pos: 0-8.
#         # 如果 pos < 8: 表示它是 Center 的某个邻居 (0,1,2,3, 5,6,7,8 -> 映射为 label 0-7)
#         # 如果 pos == 8: 表示它是 Outsider

#         target_class = random.randint(0, 8)  # 0-7: Neighbors, 8: Outsider

#         candidate_piece = None

#         if target_class == 8:
#             # Outsider: 随便选一个 guest patch
#             candidate_piece = random.choice(guest_patches)
#         else:
#             # Neighbor: 我们需要找到“真值”是 pos 的那个块
#             # target_class 对应真实的 piece ID (0,1,2,3, 5,6,7,8)
#             # 我们需要把 0-7 映射回 piece ID
#             target_piece_id = target_class
#             if target_piece_id >= 4:
#                 target_piece_id += 1

#             # 现在的 host_patches 是乱序的(scrambled)，perm_idx 告诉我们要找的 piece ID 在哪个位置
#             # perm_idx[k] == target_piece_id，那么 host_patches[k] 就是我们要的块
#             try:
#                 current_pos = perm_idx.index(target_piece_id)
#                 candidate_piece = host_patches[current_pos]
#             except ValueError:
#                 # 容错：如果找不到(极少见)，就给个全黑，label设为8
#                 candidate_piece = torch.zeros_like(center_piece)
#                 target_class = 8

#         # 归一化到 0-1 (配合 EfficientNet 和 Color Stats)
#         if center_piece.max() > 1.0:
#             center_piece /= 255.0
#         if candidate_piece.max() > 1.0:
#             candidate_piece /= 255.0

#         return (
#             center_piece,
#             candidate_piece,
#             torch.tensor(target_class, dtype=torch.long),
#         )


def train(model:nn.Module,dataloader,test_dataloader,load,epoch_num=50):
    model.train()
    loss_fn=nn.CrossEntropyLoss()
    model.to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    best_acc=0
    if load: 
        model.load_state_dict(torch.load(MODEL_PATH))
    
    for epoch in range(epoch_num):
        avg_loss=[]
        for central_piece,piece,pos in tqdm(dataloader):
            central_piece,piece,pos=central_piece.to(DEVICE),piece.to(DEVICE),pos.to(DEVICE)
            optimizer.zero_grad()
            compare_result=model(central_piece,piece)
            loss=loss_fn(compare_result,pos)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, avg_loss: {np.mean(avg_loss)}")
        if (epoch+1)%TEST_PER_EPOCH==0:
            test_acc=test(model,test_dataloader,plot=True)
            if test_acc>best_acc:
                best_acc=test_acc
                torch.save(model.state_dict(),MODEL_PATH)
            model.train()



def test(model,dataloader,plot=False):
    model.eval()
    model.to(DEVICE)
    
    total_correct = 0
    total_samples = 0

    if plot:
        confusion_matrix=torch.zeros([ACTION_NUM,ACTION_NUM])
    
    with torch.no_grad():
        for central_piece, piece, pos in tqdm(dataloader, desc="Testing"):

            central_piece = central_piece.to(DEVICE)
            piece = piece.to(DEVICE)

            compare_result = model(central_piece, piece)
            
            predictions = torch.argmax(compare_result, dim=1)

            correct = (predictions.cpu() == pos).sum().item()
            if plot:
                for pred,label in zip(predictions.cpu(),pos):
                    confusion_matrix[label,pred]+=1

            batch_size_current = central_piece.size(0)
            total_correct += correct
            total_samples += batch_size_current

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    if plot:
        category_list=list(range(0,9))
        plot_confusion_matrix(confusion_matrix,category_list,FILENAME)
    
    return accuracy


        
if __name__=="__main__":
    train_dataloader=DataLoader(DeepuzzleDataset(train_x,train_y),batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    test_dataloader=DataLoader(DeepuzzleDataset(test_x,test_y),batch_size=BATCH_SIZE)
    model=compare_model(out=ACTION_NUM).to(DEVICE)
    train(model,train_dataloader,test_dataloader,LOAD,epoch_num=EPOCH_NUM)

    model.load_state_dict(torch.load(MODEL_PATH))
    test(model,test_dataloader,True)