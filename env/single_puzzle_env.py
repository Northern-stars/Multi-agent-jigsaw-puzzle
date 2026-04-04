import torch
import numpy as np
import random
import copy
import itertools
import cv2



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
print(f"Data shape: x {train_x.shape}, y {train_y.shape}")
class Env:
    def __init__(self,
                 train_x,
                 train_y,
                 image_num,
                 buffer_size,
                 piece_num=9,
                 epsilon=0.5,
                 device= "cuda" if torch.cuda.is_available() else "cpu",
                 reward_dict={"PAIRWISE":.2,"CATE":.8,"CONSISTENCY":0,"DONE_REWARD":1000,"CONSISTENCY_REWARD":0,"PANELTY":-0.5}
                 ,epoch=5
                 ):
        self.image=train_x
        self.sample_number=train_x.shape[0]
        self.label=train_y
        self.permutation2piece={}
        self.cur_permutation={}
        # self.main_critic_model=copy.deepcopy(self.critic_model)
        self.image_num=image_num
        self.buffer_size=buffer_size
        self.action_list=[[0 for _ in range((piece_num+buffer_size)*piece_num//2+1)] for __ in range(image_num)]
        self.piece_num=piece_num
        self.permutation_list=[]
        self.reward_dict=reward_dict
        self.epsilon=epsilon
        self.device=device
        self.epochs=epoch

    
    def load_image(self,image_num,id=[]):
        if id:
            image_index=id
        else:
            image_index=random.choices(range(0,self.sample_number),k=image_num)

        for i in range(len(image_index)):
            image_raw=self.image[image_index[i]]
            permutation_raw=self.label[image_index[i]]
            image_raw=torch.tensor(image_raw).permute(2,0,1).to(torch.float)
            image_fragments=[
                image_raw[:,0:96,0:96],
                image_raw[:,0:96,96:192],
                image_raw[:,0:96,192:288],
                image_raw[:,96:192,0:96],
                image_raw[:,96:192,96:192],
                image_raw[:,96:192,192:288],
                image_raw[:,192:288,0:96],
                image_raw[:,192:288,96:192],
                image_raw[:,192:288,192:288]
            ]
            permutation_raw=list(permutation_raw)
            for m in range(8):
                for n in range(8):
                    # print(type(imageLabel[i]))
                    # print(imageLabel)
                    if permutation_raw[m][n]==True:
                        if n>=4:
                            permutation_raw[m]=n+1
                            break
                        else:
                            permutation_raw[m]=n
                            break
            permutation_raw.insert(4,4)
            for j in range(9):
                self.permutation2piece[permutation_raw[j]+9*i]=image_fragments[j]
            self.permutation2piece[-1]=torch.zeros(3,96,96)
        
    def get_image(self,permutation,image_index):
        image=torch.zeros(3,288,288)
        final_permutation=copy.deepcopy(permutation)
        final_permutation.insert(9//2,image_index*9+9//2)
        for i in range(9):
            image[:,(0+i//3)*96:(1+i//3)*96,(0+i%3)*96:(1+i%3)*96]=self.permutation2piece[final_permutation[i]]
    
        # outsider_piece=self.permutation2piece[permutation[-1]]
        # return image.unsqueeze(0).to(self.device),outsider_piece.unsqueeze(0).to(self.device)
        return image.unsqueeze(0).to(self.device),self.permutation2piece[-1].to(self.device)
    def request_for_image(self,image_id,permutation,image_index):
        self.load_image(image_num=self.image_num,id=image_id)
        image,outsider=self.get_image(permutation=permutation,image_index=image_index)
        self.load_image(image_num=self.image_num,id=self.image_id)
        return image,outsider
    
    def get_reward(self,permutation_list):
        """Return: local reward list, consistency reward list, done list, consistency list"""
        # permutation_list=permutation_list[::][:len(permutation_list[0])-1]#Comment if the buffer is not after the permutation
        permutation_copy=copy.deepcopy(permutation_list)
        for i in range(len(permutation_list)):
            permutation_copy[i].insert(self.piece_num//2,i*self.piece_num+self.piece_num//2)
        done_list=[0 for i in range(len(permutation_copy))]
        consistency_list=[0 for i in range(len(permutation_copy))]
        local_reward_list=[0 for i in range(len(permutation_copy))]
        consistency_reward_list=[0 for i in range(len(permutation_copy))]
        edge_length=int(len(permutation_copy[0])**0.5)
        piece_num=len(permutation_copy[0])
        hori_set=[(i,i+1) for i in [j for j in range(piece_num) if j%edge_length!=edge_length-1 ]]
        vert_set=[(i,i+edge_length) for i in range(piece_num-edge_length)]
        
        for i in range(len(permutation_list)):
            for j in range(len(hori_set)):#Pair reward

                hori_pair_set=(permutation_copy[i][hori_set[j][0]],permutation_copy[i][hori_set[j][1]])
                vert_pair_set=(permutation_copy[i][vert_set[j][0]],permutation_copy[i][vert_set[j][1]])
                if (-1 not in hori_pair_set) and (hori_pair_set[0]%piece_num,hori_pair_set[1]%piece_num) in hori_set and (hori_pair_set[0]//piece_num==hori_pair_set[1]//piece_num)and(hori_pair_set[0]//piece_num==i):
                    local_reward_list[i]+=1*self.reward_dict["PAIRWISE"]
                if (-1 not in vert_pair_set) and (vert_pair_set[0]%piece_num,vert_pair_set[1]%piece_num) in vert_set and (vert_pair_set[0]//piece_num==vert_pair_set[1]//piece_num)and (vert_pair_set[0]//piece_num==i):
                    local_reward_list[i]+=1*self.reward_dict["PAIRWISE"]

            piece_range=[0 for j in range (len(permutation_list))]
            # print(piece_range)
        
            for j in range(piece_num):
                if permutation_copy[i][j]!=-1:
                    piece_range[permutation_copy[i][j]//piece_num]+=1
                    local_reward_list[i]+=(permutation_copy[i][j]%piece_num==j and permutation_copy[i][j]//piece_num==i)*self.reward_dict["CATE"]#Category reward
            
            
            max_piece=piece_range[i]#Consistancy reward

            
            # if -1 in permutation_copy[i]:
            #     consistency_reward_list[j]+=0.5*CONSISTENCY_REWARD
            consistency_reward_list[i]=max_piece*self.reward_dict["CONSISTENCY"]
            if max_piece==piece_num:
                consistency_reward_list[i]=self.reward_dict["CONSISTENCY_REWARD"]
                consistency_list[i]=True

            local_reward_list[i]+=self.reward_dict["PANELTY"]
            consistency_reward_list[i]+=self.reward_dict["PANELTY"]
            start_index=min(permutation_copy[i])//piece_num*piece_num#Done reward
            if permutation_copy[i]==list(range(start_index,start_index+piece_num)):
                done_list[i]=True
                local_reward_list[i]=self.reward_dict["DONE_REWARD"]
        return local_reward_list,consistency_reward_list,done_list,consistency_list
        

    def clean_memory(self):
        self.mkv_memory=[]
        self.memory_counter=0

    def show_image(self,image_permutation_list):
        for i in range(self.image_num):

            image=self.get_image(permutation=image_permutation_list[i],image_index=i)
            image=image.squeeze().to("cpu")
            image=image.permute([1,2,0]).numpy().astype(np.uint8)
            cv2.imshow(f"Final image {i}",image)
        cv2.waitKey(1)
        # time.sleep(10)
        # cv2.destroyAllWindows()
    
    def permute(self,cur_permutation,action_index):
        new_permutation=copy.deepcopy(cur_permutation)
        if action_index==(self.piece_num+1)*self.piece_num//2:
            return new_permutation
        action=list(itertools.combinations(list(range(len(cur_permutation))), 2))[action_index]
        value0=cur_permutation[action[0]]
        value1=cur_permutation[action[1]]
        new_permutation[action[0]]=value1
        new_permutation[action[1]]=value0
        return new_permutation
    

    def summon_permutation_list(self,swap_num,id=[]):
        # print("Summon initial permutation")
            if id:
                image_index=id
            else:
                image_index=random.choices(range(0,self.sample_number),k=self.image_num)
            self.image_id=image_index
            self.load_image(image_num=self.image_num,id=self.image_id)
            print(f"Episode image:{self.image_id}")
            initial_permutation=list(range(self.piece_num*self.image_num))
            for i in range(self.image_num):
                initial_permutation.pop(9*i+9//2-i)
            for i in range(swap_num):
                action_index=random.randint(0,len(initial_permutation)*(len(initial_permutation)-1)//2-1)
                initial_permutation=self.permute(initial_permutation,action_index)
            print(f"Initial permutation {initial_permutation}")
            self.permutation_list=[initial_permutation[j*(self.piece_num-1):(j+1)*(self.piece_num-1)]
                                    for j in range(self.image_num)]
            