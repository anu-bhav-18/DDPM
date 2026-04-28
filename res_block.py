# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import torch
import torch.nn as nn

class ResBlock(nn.Module):
  def __init__(self,in_channnels,out_channel,time_embedding_dim,num_group,device=None):
    super(ResBlock,self).__init__()

    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.norm1 = nn.GroupNorm(num_groups=num_group,num_channels=in_channnels)
    self.act_fn_1 = nn.SiLU()
    self.conv1  = nn.Conv2d(in_channels=in_channnels,out_channels=out_channel,kernel_size=3,stride=1,padding=1,)

    self.time_proj = nn.Linear(in_features=time_embedding_dim,out_features=out_channel)

    self.norm2 =nn.GroupNorm(num_groups=num_group,num_channels=out_channel)
    self.act_fn_2 = nn.SiLU()
    self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1)

    self.residaul = nn.Conv2d(in_channnels,out_channel,kernel_size=1) if in_channnels!=out_channel else nn.Identity()



  def forward(self,x,t):
    # shpae of x (B,C,H,W)
    # shape of t (B,d_model==embedding_dim)
    #print(f"Shape of x before assining:-{x.shape}")
    h = x
    x = self.norm1(x)
    x = self.act_fn_1(x)
    x = self.conv1(x)
    #print(f"shape after first layer:-{x.shape}")

    #t = self.time_proj(t)
    #print(f"T shape before :-{t.shape}")
    t = self.time_proj(t)[:, :, None, None]
    #print(f"Shape After unsqueeze:-{t.shape}")
    #print(f"Shape of x:-{x.shape}")
    x = x+t
    #print(f"shape after adding the time embedding:-{x.shape}\ntime :-{t.shape}")

    x = self.norm2(x)
    x = self.act_fn_2(x)
    x = self.conv2(x)
    #print(f"Shape of x:-{x.shape}")
    #print(f"Shape of H:-{h.shape}")
    h = self.residaul(h)
    #print(f"shape of h after giving to residual:-{h.shape}")

    return x+h
