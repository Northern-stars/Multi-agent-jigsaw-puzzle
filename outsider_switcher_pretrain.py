import torch.nn as nn
import torch
from torchvision.models import efficientnet_b0 ,efficientnet_b3
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import hamming_loss, f1_score


DEVICE="cuda" if torch.cuda.is_available() else "cpu"

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

BATCH_SIZE=80
TEST_PER_EPOCH=20
MODEL_NAME="outsider_switcher_pretrain.pth"

class fen_model(nn.Module):
    def __init__(self, hidden_size1, hidden_size2):
        super(fen_model, self).__init__()
        self.ef = efficientnet_b3(weights="DEFAULT")
        self.ef.classifier = nn.Linear(1536, 1024)

        self.contrast_fc_hori = nn.Linear(2048, 1024)
        self.contrast_fc_vert = nn.Linear(2048, 1024)

        self.fc1 = nn.Linear(1024*12, hidden_size1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size1)
        self.do = nn.Dropout1d(p=0.1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 定义 index 对
        self.hori_set = [(i, i+1) for i in range(9) if i % 3 != 2]
        self.vert_set = [(i, i+3) for i in range(6)]

    def forward(self, image):
        B, C, H, W = image.shape   # 假设输入 (B, 3, 288, 288)

        # ---- 1. 切片并 reshape 成 batch ----
        patches = image.unfold(2, 96, 96).unfold(3, 96, 96)
        # patches: (B, C, 3, 3, 96, 96)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        patches = patches.view(B*9, C, 96, 96)   # (B*9, 3, 96, 96)

        # ---- 2. 一次性送进 ef ----
        feats = self.ef(patches)   # (B*9, 512)
        feats = feats.view(B, 9, 1024)  # (B, 9, 512)

        # ---- 3. 构建横向对 ----
        hori_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.hori_set], dim=1)  # (B, #pairs, 1024)
        hori_feats = self.contrast_fc_hori(hori_pairs)  # (B, #pairs, 512)

        # ---- 4. 构建纵向对 ----
        vert_pairs = torch.stack([torch.cat([feats[:, i, :], feats[:, j, :]], dim=-1) 
                                  for (i,j) in self.vert_set], dim=1)  # (B, #pairs, 1024)
        vert_feats = self.contrast_fc_vert(vert_pairs)  # (B, #pairs, 512)

        # ---- 5. 拼接所有特征 ----
        feature_tensor = torch.cat([hori_feats, vert_feats], dim=1)  # (B, 12, 512)
        feature_tensor = feature_tensor.view(B, -1)   # (B, 12*512)

        # ---- 6. 全连接部分 ----
        x = self.do(feature_tensor)
        x = self.fc1(x)
        x = self.do(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x




class actor_model(nn.Module):
    def __init__(self,hidden_size1,hidden_size2,outsider_hidden_size,action_num):
        super(actor_model,self).__init__()
        self.fen_model=fen_model(hidden_size1,hidden_size1)
        # state_dict=torch.load("pairwise_pretrain.pth")
        # state_dict_replace = {
        # k: v 
        # for k, v in state_dict.items() 
        # if k.startswith("ef.")
        # }
        # load_result_hori=self.image_fen_model.load_state_dict(state_dict_replace,strict=False)
        # print("Actor missing keys hori",load_result_hori.missing_keys)
        # print("Actor unexpected keys hori",load_result_hori.unexpected_keys)
        self.outsider_fen_model=efficientnet_b0(weights="DEFAULT")
        self.outsider_fen_model.classifier=nn.Linear(1280,outsider_hidden_size)
        self.outsider_contrast_fc=nn.Linear(2*outsider_hidden_size,outsider_hidden_size)
        self.outsider_fc=nn.Linear(outsider_hidden_size,outsider_hidden_size)
        self.fc1=nn.Linear(hidden_size1+outsider_hidden_size,hidden_size2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.1)
        self.bn=nn.BatchNorm1d(hidden_size2)
        self.fc2=nn.Linear(hidden_size2,action_num)
    
    def forward(self,image,outsider_piece):
        # image_fragments=[
        #     image[:,:,0:96,0:96],
        #     image[:,:,0:96,96:192],
        #     image[:,:,0:96,192:288],
        #     image[:,:,96:192,0:96],
        #     image[:,:,96:192,96:192],
        #     image[:,:,96:192,192:288],
        #     image[:,:,192:288,0:96],
        #     image[:,:,192:288,96:192],
        #     image[:,:,192:288,192:288]
        # ]
        central_image=image[:,:,96:192,96:192]
        image_input=self.fen_model(image)
        outsider_input=self.outsider_fen_model(outsider_piece)
        # outsider_image_tensor=[self.outsider_fen_model(image_fragments[i]) for i in range(len(image_fragments)) ]
        outsider_image_tensor=self.outsider_fen_model(central_image)
        outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,outsider_image_tensor],dim=-1)) 
        # outsider_tensor=self.outsider_contrast_fc(torch.cat([outsider_input,image_input],dim=-1))
        outsider_tensor=self.outsider_fc(outsider_tensor)
        feature_tensor=torch.cat([image_input,outsider_tensor],dim=-1)
        out=self.fc1(feature_tensor)
        out=self.dropout(out)
        out=self.bn(out)
        out=self.relu(out)
        out=self.fc2(out)
        # out=nn.functional.sigmoid(out)
        return out
    

class imageData(Dataset):
    def __init__(self,data,label,swap_num=None):
        self.data=data
        self.label=label
        self.swap_num=swap_num
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        outsider_index=index
        while outsider_index==index:
            outsider_index=random.randint(0,len(self.data)-1)
        if self.swap_num is None:
            swap_num=random.randint(1,4)
        else:
            swap_num=self.swap_num
        image=torch.tensor(self.data[index]).permute([2,0,1]).to(torch.float)
        outsider_image=torch.tensor(self.data[outsider_index]).permute([2,0,1]).to(torch.float)
        imageLabel=list(self.label[index])
        outsiderImageLabel=list(self.label[outsider_index])
        for i in range(8):
            for j in range(8):
                if imageLabel[i][j]==True:
                    if j>=4:
                        imageLabel[i]=j+1
                        break
                    else:
                        imageLabel[i]=j
                        break
        for i in range(8):
            for j in range(8):
                if outsiderImageLabel[i][j]==True:
                    if j>=4:
                        outsiderImageLabel[i]=j+1
                        break
                    else:
                        outsiderImageLabel[i]=j
                        break
        imageLabel.insert(4,4)
        outsiderImageLabel.insert(4,4)
        image_fragments=[image[:,(imageLabel[i]//3)*96:(imageLabel[i]//3+1)*96,(imageLabel[i]%3)*96:(imageLabel[i]%3+1)*96] for i in range(9)]
        outsider_image_fragments=[outsider_image[:,(outsiderImageLabel[i]//3)*96:(outsiderImageLabel[i]//3+1)*96,(outsiderImageLabel[i]%3)*96:(outsiderImageLabel[i]%3+1)*96] for i in range(9)]  
        
        outsider_pieces=[]
        label_list=[]

        central_outsider_piece=outsider_image_fragments[4]
        outsider_image_fragments.pop(4)
        random.shuffle(outsider_image_fragments)
        outsider_image_fragments.insert(4,central_outsider_piece) # shuffle the outsider

        central_piece=image_fragments[4]
        image_fragments.pop(4)
        random.shuffle(image_fragments)
        image_fragments.insert(4,central_piece)
        #shuffle the rest 8 pieces

        one_hot_label=[0 for _ in range(8)]

        for i in range(swap_num):
            random_index=random.randint(0,len(image_fragments)-2)
            if random_index not in label_list:
                one_hot_label[random_index]=1
                label_list.append(random_index)
                if random_index>=4:
                    random_index+=1
                outsider_pieces.append(image_fragments[random_index])
                image_fragments[random_index]=outsider_image_fragments[random_index]

        

        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=image_fragments[i]
        
        # label=random.randint(0,len(outsider_pieces))
        # if label==len(outsider_pieces):
        #     outsider_piece=random.choice(outsider_image_fragments)
        # else:
        #     outsider_piece=outsider_pieces[label]
        #     label=label_list[label]


        # label=random.randint(0,8)
        # if label==8:
        #     outsider_piece=random.choice(outsider_image_fragments)
        # else:
        #     if label>=4:
        #         label+=1
        #     outsider_piece=image_fragments[label]
        
        outsider_piece=random.choice(outsider_pieces)



        return image,outsider_piece,torch.tensor(one_hot_label).to(torch.float)


model=actor_model(hidden_size1=2048,
                  hidden_size2=1024
                  ,outsider_hidden_size=1024
                  ,action_num=8).to(DEVICE)
# loss_fn=nn.CrossEntropyLoss()
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-8)

train_data=imageData(data=train_x,label=train_y)
train_dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
test_data=imageData(test_x,test_y)
test_dataloader=DataLoader(test_data,batch_size=1,shuffle=True)


def train(epoch_num=500, load=True):
    if load:
        model.load_state_dict(torch.load(MODEL_NAME))
    print("Start training")
    for epoch in range(epoch_num):
        loss_sum=0
        model.train()
        for batch_num,(image,outsider_piece,label) in enumerate(train_dataloader):
            # print(image.size())
            image,outsider_piece,label=image.to(DEVICE),outsider_piece.to(DEVICE),label.to(DEVICE)
            output=model(image,outsider_piece)
            loss=loss_fn(output,label)
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"epoch: {epoch}, loss_sum: {loss_sum*BATCH_SIZE/len(train_data)}")
    
        print("Saving model")
        torch.save(model.state_dict(),MODEL_NAME)

        if epoch%TEST_PER_EPOCH==TEST_PER_EPOCH-1:
            test()
            
    



def test():
    print("start testing")
    model.load_state_dict(torch.load(MODEL_NAME))
    model.eval()
    all_preds=[]
    all_labels=[]
    accuracy=[]
    loss_sum=0
    total_accuracy=[]
    with torch.no_grad():
        for batch_num,(image,outsider_piece,label) in enumerate(test_dataloader):
            image,outsider_piece,label=image.to(DEVICE),outsider_piece.to(DEVICE),label.to(DEVICE)
            probs=model(image,outsider_piece)
            preds=(probs>0.5).float()
            loss=loss_fn(probs,label)
            loss_sum+=loss.item()
            # pred_label=torch.argmax(probs,dim=1)
            # if pred_label.item()==label.item():
            #     total_accuracy.append(1)
            # else:
            #     total_accuracy.append(0)
            bit_accuracy = (preds == label).float().mean().item()
            accuracy.append(bit_accuracy)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            # print((preds==label).all())
            total_accuracy.append((preds==label).cpu().all())
        
        print(f"Loss: {loss_sum/len(test_data)}")

        all_preds=np.vstack(all_preds)
        all_labels=np.vstack(all_labels)

        h_loss=hamming_loss(all_labels,all_preds)

        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        print(f"Hamming loss: {h_loss}, F1 macro: {f1_macro}, F1 micro: {f1_micro}, Accuracy: {np.mean(accuracy)}, Total accuracy: {np.mean(total_accuracy)}")
        print(f"Total accuracy: {np.mean(total_accuracy)}")

if __name__=="__main__":
    train(epoch_num=100,load=False)
    test()