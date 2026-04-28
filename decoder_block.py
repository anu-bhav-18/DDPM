# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, num_group,use_atten=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim,num_group)
        self.res2 = ResBlock(out_ch, out_ch, time_dim,num_group)
        self.attn = _SelfAttention(out_ch,num_group) if use_atten else nn.Identity()
        self.up = Upsample(out_ch)

    def forward(self, x, skip, t):
        #print(f"Shape of x:-{x.shape}")
        #print(f"shape of skip:-{skip.shape}")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        x = self.up(x)
        return x