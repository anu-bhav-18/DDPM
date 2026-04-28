# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from torch.distributed import is_available
class EncoderBlock(nn.Module):
  def __init__(self,in_channel,out_channel,time_dim,num_group , use_atten= False ,device = None ):
    super(EncoderBlock, self).__init__()

    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.res_1  = ResBlock(in_channel,out_channel,time_dim,num_group)
    self.res_2 = ResBlock(out_channel,out_channel,time_dim,num_group,self.device)
    self.atten = _SelfAttention(out_channel,num_group) if use_atten else nn.Identity()
    self.down  = Downsample(out_channel)


  def forward(self,x,t):
    x = self.res_1(x,t)
    x = self.res_2(x,t)
    x = self.atten(x)
    skip = x
    x = self.down(x)
    #print(f"Skip:-{skip.shape}\n shpae of x :-{x.shape}")
    return x,skip
