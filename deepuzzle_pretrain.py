import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from piece_compare import piece_compare_model_b3 as compare_model
import random
import numpy as np
import os
from utils import plot_confusion_matrix

LOAD=False
MODEL_PATH=os.path.join("model","deepuzzle.pth")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TEST_PER_EPOCH=20
BATCH_SIZE=150
EPOCH_NUM=250
FILENAME="_deepuzzle_9"


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

        pos=random.randint(0,8)
        if pos==8:
            piece=random.choice(image2_fragments)
        elif pos>=4:
            piece=image1_fragments[image1_label.index(pos+1)]
        else:
            piece=image1_fragments[image1_label.index(pos)]
        

        return central_piece,piece,pos


def train(model:nn.Module,dataloader,test_dataloader,load,epoch_num=50):
    model.train()
    loss_fn=nn.CrossEntropyLoss()
    model.to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    if load:
        model.load_state_dict(torch.load(MODEL_PATH))
    
    for epoch in range(epoch_num):
        avg_loss=[]
        for central_piece,piece,pos in tqdm(dataloader):
            compare_result=model(central_piece.to(DEVICE),piece.to(DEVICE))
            loss=loss_fn(compare_result,pos.to(DEVICE))
            avg_loss.append(loss.item())
            total_correct += (compare_result.argmax(1) == pos).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, avg_loss: {np.mean(avg_loss)}")
        torch.save(model.state_dict(),MODEL_PATH)
        if (epoch+1)%TEST_PER_EPOCH==0:
            test(model,test_dataloader)
            model.train()



def test(model,dataloader,plot=False):
    model.eval()
    model.to(DEVICE)
    
    total_correct = 0
    total_samples = 0

    if plot:
        confusion_matrix=torch.zeros([9,9])
    
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
    
    # return accuracy


        
if __name__=="__main__":
    train_dataloader=DataLoader(DeepuzzleDataset(train_x,train_y),batch_size=BATCH_SIZE,drop_last=True)
    test_dataloader=DataLoader(DeepuzzleDataset(test_x,test_y),batch_size=BATCH_SIZE)
    model=compare_model(512,512).to(DEVICE)
    # train(model,train_dataloader,test_dataloader,LOAD,epoch_num=EPOCH_NUM)

    model.load_state_dict(torch.load(MODEL_PATH))
    test(model,test_dataloader,True)