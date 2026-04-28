# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
class Bottleneck(nn.Module):
    def __init__(self, channels, time_dim,num_group):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_dim,num_group)
        self.attn = _SelfAttention(channels,num_group)
        self.res2 = ResBlock(channels, channels, time_dim,num_group)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x