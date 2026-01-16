import torch 
from torch import nn
import torch.nn.functional as F


class gated_linear(nn.Module):

    def __init__(self, in_features,out_features,bias=True):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.bias=bias

        self.weightA=nn.Parameter(torch.Tensor(out_features,in_features))
        self.weightB=nn.Parameter(torch.Tensor(out_features,in_features))

        if bias:
            self.biasA=nn.Parameter(torch.Tensor(out_features))
            self.biasB=nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("biasA",None)
            self.register_parameter("biasB",None)
        self.gate=nn.Parameter(torch.Tensor(out_features,in_features))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weightA,a=5**0.5)
        nn.init.kaiming_uniform_(self.weightB,a=5**0.5)
        if self.bias:
            fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weightA)
            bound=1/(fan_in**0.5)
            nn.init.uniform_(self.biasA,-bound,bound)
            nn.init.uniform_(self.biasB,-bound,bound)
        nn.init.kaiming_uniform_(self.gate,a=5**0.5)
    
    def forward(self,x):
        batch_size=x.size(0)
        out_a=F.linear(x,self.weightA,self.biasA)
        out_b=F.linear(x,self.weightB,self.biasB)

        gate_tensor=F.linear(x,self.gate)

        # batched_gate=self.gate.unsqueeze(0).expand(batch_size,-1)

        # condition=(x>batched_gate)
        condition=(gate_tensor>0.5)
        out=torch.where(condition,out_a,out_b)
        return out

class gated_linear_gumble_sigmoid(nn.Module):

    def __init__(self, in_features,out_features,bias=True):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.bias=bias


        self.weightA=nn.Parameter(torch.Tensor(out_features,in_features))
        self.weightB=nn.Parameter(torch.Tensor(out_features,in_features))
        self.gate=nn.Parameter(torch.Tensor(out_features,in_features))

        if bias:
            self.biasA=nn.Parameter(torch.Tensor(out_features))
            self.biasB=nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("biasA",None)
            self.register_parameter("biasB",None)
        

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weightA,a=5**0.5)
        nn.init.kaiming_uniform_(self.weightB,a=5**0.5)
        nn.init.kaiming_uniform_(self.gate,a=5**0.5)
        if self.bias:
            fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weightA)
            bound=1/(fan_in**0.5)
            nn.init.uniform_(self.biasA,-bound,bound)
            nn.init.uniform_(self.biasB,-bound,bound)
    
    def sample_gumbel(self,shape,eps=1e-10):
        U=torch.rand(shape)
        return -torch.log(-torch.log(U+eps)+eps)
    
    def gumbel_sigmoid(self,logit,tau=1):
        g=self.sample_gumbel(logit.shape).to(logit.device)
        return torch.sigmoid((logit+g)/tau)
    

    def forward(self,x,tau=1):
        batch_size=x.size(0)
        out_a=F.linear(x,self.weightA,self.biasA)
        out_b=F.linear(x,self.weightB,self.biasB)

        gate_logits=F.linear(x,self.gate)


        if self.training:
            gate_val=self.gumbel_sigmoid(gate_logits,tau=tau)
        else:
            gate_val=torch.sigmoid(gate_logits)
            gate_val=(gate_val>0.5).float()
        out=gate_val*out_a+(1-gate_val)*out_b

        return out
