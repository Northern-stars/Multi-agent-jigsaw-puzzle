import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
#Picture:[batchsize,channel,width,height]

class UNet(nn.Module):
    def __init__(self, input_channel,output_channel,hidden):
        super().__init__()
        self.relu=nn.ReLU()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        self.e11 = nn.Conv2d(input_channel, hidden//16, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(hidden//16, hidden//16, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(hidden//16, hidden//8, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(hidden//8, hidden//8, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(hidden//8, hidden//4, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(hidden//4, hidden//4, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(hidden//4, hidden//2, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(hidden//2, hidden//2, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(hidden//2, hidden, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(hidden, hidden//2, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(hidden, hidden//2, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(hidden//2, hidden//2, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(hidden//2, hidden//4, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(hidden//2, hidden//4, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(hidden//4, hidden//4, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(hidden//4, hidden//8, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(hidden//4, hidden//8, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(hidden//8, hidden//8, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(hidden//8, hidden//16, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(hidden//8, output_channel, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)


    def forward(self, x):
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.relu(self.e41(xp3))
        xe42 = self.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.relu(self.e51(xp4))
        xe52 = self.relu(self.e52(xe51))
                
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        # print(xu4.size(),xe12.size())
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.relu(self.d41(xu44))
        xd42 = self.relu(self.d42(xd41))

        return xd42

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,num_patches):
        super(PositionalEmbedding,self).__init__()
        
        self.length=num_patches+1
        self.position_embedding_code=nn.Parameter(torch.randn(1,self.length,d_model))
        nn.init.normal_(self.position_embedding_code)
    
    def forward(self,x):
        for i in range(9):
            x[:,self.length*i:self.length*(i+1),:]+=i+self.position_embedding_code
        # x=x+self.position_embedding_code
        return x

class PictureEmbedding(nn.Module):
    def __init__(self,picture_size,patch_size):
        super(PictureEmbedding,self).__init__()
        self.width=picture_size[2]
        self.height=picture_size[3]
        self.channel_num=picture_size[1]
        self.batch_size=picture_size[0]
        self.patch_size=patch_size
        self.class_token=nn.Parameter(torch.randn(1,1,self.channel_num*patch_size**2))
        nn.init.normal_(self.class_token)

    def forward(self,x):
        with torch.no_grad():
            # print(x.size())
            batch_size=x.size(0)
            assert self.width % self.patch_size == 0 and self.height % self.patch_size == 0
            patches=F.unfold(x,kernel_size=self.patch_size,stride=self.patch_size)
            patched_image=patches.permute(0,2,1)
            class_token=self.class_token.repeat(batch_size,1,1)
            patched_image=torch.cat([patched_image,class_token],dim=1)
        return patched_image

class PictureEncoding(nn.Module):
    def __init__(self,picture_size,channel,patch_size,drop_prob):
        super(PictureEncoding,self).__init__()
        self.tok_emb=PictureEmbedding(picture_size,patch_size)
        self.pos_emb=PositionalEmbedding(d_model=channel*(patch_size**2),num_patches=picture_size[2]*picture_size[3]//(patch_size**2))
        self.drop_out=nn.Dropout(p=drop_prob)
        self.patch_size=patch_size
    def forward(self,x):
        with torch.no_grad():#changed
            batch_size,width,height,channel=x.size()
            x=self.tok_emb(x)
            out=self.pos_emb(x)
        return out


        
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head# the number of head
        self.d_model=d_model
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_combine=nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,q,k,v,mask=None):
        # print("q:",q.size())
        batch,time,dimension=q.size()
        n_d=self.d_model//self.n_head
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        # print("q=",q.size())
        q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score=torch.matmul(q,k.transpose(2,3))/math.sqrt(n_d)
        if mask is not None:
            score=score.masked_fill(mask==0,-10000)#mask code
        score=self.softmax(score)@v# @: matrix multiplication
        score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension)
        output=self.w_combine(score)
        return output

class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps#stability,extremely small number
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        # var=x.var(-1,unbiased=False,keepdim=True)
        var=x.var(-1,unbiased=True,keepdim=True)

        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta
        return out
    
class PositionwiseForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositionwiseForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x



class EncoderLayer(nn.Module):#single Encoder layer
    def __init__(self,d_model,hidden,n_head,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head)
        self.layernorm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=PositionwiseForward(d_model,hidden,dropout)
        self.layernorm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        _x=x #save the origin input into _x,for calculating residual shrinkage
        # print('Attention:',x.size())
        x=self.attention(x,x,x,mask)#let the x be the q,k,v and calculate the attention score
        x=self.dropout1(x)
        x=self.layernorm1(x+_x) #residual shrinkage link
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.layernorm2(x+_x)
        return x


class Encoder(nn.Module):
    def __init__(self,picture_size,channel,patch_size,hidden,n_head,n_layer,dropout=0.1):
        super(Encoder,self).__init__()
        self.embedding=PictureEncoding(picture_size,channel,patch_size,dropout)
        self.layers=nn.ModuleList( 
            [
                EncoderLayer(patch_size*patch_size*channel,hidden,n_head)
                for _ in range(n_layer)#create n_layer Encoderlayers
            ]
        )

    def forward(self,x,s_mask=None):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_mask) #pass
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 picture_size,
                 patch_size,
                 encoder_layer_num,
                 n_head,
                 out_size,
                 output_channel,
                 unet_hidden,
                 encoder_hidden,
                 dropout=0.1):
        super().__init__()
        # self.local_fen=UNet(input_channel=picture_size[1],output_channel=output_channel,hidden=unet_hidden)
        self.encoder=Encoder(picture_size=picture_size,patch_size=patch_size,channel=output_channel,hidden=encoder_hidden,n_layer=encoder_layer_num,n_head=n_head,dropout=dropout)
        self.out_layer=nn.Linear(output_channel*(patch_size**2),out_size)

    
    def forward(self,x):

        # x=self.local_fen(x)
        x=self.encoder(x)
        x=x[:,-1,:]

        out=self.out_layer(x)
        return out

if __name__=="__main__":
    tensor=torch.randn(5,3,288,288).to(DEVICE)
    model=VisionTransformer(
        picture_size=[5,3,288,288],
        patch_size=16,
        encoder_hidden=512,
        out_size=1280,
        n_head=4,
        encoder_layer_num=2,
        unet_hidden=1024,
        output_channel=6
    ).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-8)
    answer=model(tensor)
    loss_fn=nn.MSELoss()
    true_label=torch.randn(5,1280).to(DEVICE)
    print(answer.size())
    loss=loss_fn(answer,true_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

        
