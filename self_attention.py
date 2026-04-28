
# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import torch
import torch.nn as nn

class _SelfAttention(nn.Module):
  def __init__(self,in_channels,num_groups,device =None):
    super(_SelfAttention,self).__init__()
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.norm1 = nn.GroupNorm(num_groups=num_groups,num_channels=in_channels)

    self.q_w  = nn.Conv2d(in_channels,in_channels,kernel_size=1)
    self.k_w = nn.Conv2d(in_channels,in_channels,kernel_size=1)
    self.v_w = nn.Conv2d(in_channels,in_channels,kernel_size=1)

    self.proj = nn.Conv2d(in_channels,in_channels,kernel_size=1)

  def forward(self,x):
    #shape of inuput x(batch_size,channel,height,width)
    h = x
    #print(f"Before assining x shape:-{h.shape}")
    x = self.norm1(x)

    Q = self.q_w(x)
    #print(f"Q shape:-{Q.shape}")
    K = self.k_w(x)
    #print(f"K shape:-{K.shape}")
    V = self.v_w(x)
    #print(f"V shape:-{V.shape}")

    B,C,H,W = x.shape

    Q = Q.reshape(B,Q.size(1),H*W)
    #print(f"Q shape:-{Q.shape}")
    K = K.reshape(B,K.size(1),H*W)
    #print(f"K shape:-{K.shape}")
    V = V.reshape(B,V.size(1),H*W)
    #print(f"V shape:-{V.shape}")

    atten_score  = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(V.size(1))
    #print(f"Attention score dim:-{atten_score.shape}")
    atten_score = torch.softmax(atten_score,dim=-1)
    #print(f"Shape after softmax:-{atten_score.shape}")
    context  = torch.matmul(atten_score,V)
  #  print(f"Context mat shape:- {context.shape}")

    output = context.reshape(B,C,H,W)
    #print(f"Output shape:-{output.shape}")
    

    return h + self.proj(output)



