# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, time_dim=512,num_group=8):
        super().__init__()

        self.time_emd =TimeEmbedding(512, 1000)

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.enc1 = EncoderBlock(base_channels, 128, time_dim,num_group)
        self.enc2 = EncoderBlock(128, 256, time_dim,num_group)
        self.enc3 = EncoderBlock(256, 512, time_dim,num_group, use_atten=True)

        # Bottleneck
        self.bottleneck = Bottleneck(512, time_dim,num_group)

        # Decoder
        self.dec3 = DecoderBlock(512 + 512, 256, time_dim,num_group, use_atten=True)
        self.dec2 = DecoderBlock(256 + 256, 128, time_dim,num_group)
        self.dec1 = DecoderBlock(128 + 128, 128, time_dim,num_group)

        self.final_conv = nn.Conv2d(128, in_channels, 1)

    def forward(self, x, t):
        t =self.time_emd(t)
        x = self.init_conv(x)

        x1, skip1 = self.enc1(x, t)
        x2, skip2 = self.enc2(x1, t)
        x3, skip3 = self.enc3(x2, t)

        x = self.bottleneck(x3, t)

        x = self.dec3(x, skip3, t)
        x = self.dec2(x, skip2, t)
        x = self.dec1(x, skip1, t)

        return self.final_conv(x)