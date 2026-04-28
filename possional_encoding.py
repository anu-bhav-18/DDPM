# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import torch
import torch.nn as nn
import math

class PossionalEncoding(nn.Module):
  def __init__(self,d_model,time_steps,device = None):
    super(PossionalEncoding,self).__init__()
    self.d_model= d_model # a integer expected output possional encoding dim
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.times_steps = time_steps ## shape(batch_size,) total number of item for possional encoding

    self.pe = torch.zeros(self.times_steps,self.d_model,device=self.device)
    for pos in range(self.times_steps):
      for  idx in range(0,self.d_model,2):
        self.pe[pos,idx] = math.sin(pos/(10000**(2*idx/self.d_model)))
        self.pe[pos,idx+1] = math.cos(pos/(10000**(2*idx/self.d_model)))


  def forward(self,t):
    return self.pe[t]