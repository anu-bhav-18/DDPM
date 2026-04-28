# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
  def __init__(self,d_model,time_steps,device=None,hidden_dim = None):
    super(TimeEmbedding,self).__init__()
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.d_model = d_model
    self.time_step = time_steps
    self.hidden_dim = hidden_dim if hidden_dim is not None else d_model*4

    self.possional_encoding = PossionalEncoding(self.d_model,self.time_step,self.device)
    self.layer_1 = nn.Linear(self.d_model,self.hidden_dim)
    self.activation_fn = nn.SiLU()
    self.layer_2 = nn.Linear(self.hidden_dim,self.d_model)

  def forward(self,t):
    t = self.possional_encoding(t)
    t = self.layer_1(t)
    t = self.activation_fn(t)
    time_embedding = self.layer_2(t)
    return time_embedding


