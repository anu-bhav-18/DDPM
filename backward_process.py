# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt

class BackwardProcess():
  def __init__(self,num_total_steps =1000,beta_start =0.0001,beta_end = 0.02,device= None):
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.total_time_step = num_total_steps
    self.beta_start= beta_start
    self.beta_end = beta_end
    self.betas = torch.linspace(start=self.beta_start,end=self.beta_end,steps=self.total_time_step,device=self.device)
    self.alpha = 1- self.betas
    self.alphas_cumprod = torch.cumprod(self.alpha)
    self.sqrt_alpha_bar = math.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alpha_bar = math.sqrt(1-self.alphas_cumprod)


  def plot_images(self, images,t):
    batch_size = images.shape[0]
    grid_size = math.ceil(math.sqrt(batch_size))
    t = t.detach().cpu().numpy()

    fig, axes = plt.subplots(
        ncols=grid_size, 
        nrows=grid_size, 
        figsize=(grid_size * 2, grid_size * 2), 
        dpi=150
    )
    
    images = (images.detach().cpu() + 1.0) / 2.0  
    images = images.clamp(0, 1)
    images = images.permute(0, 2, 3, 1).numpy()

    axes = axes.flatten()
    for i in range(len(axes)):
        if i < batch_size:
            axes[i].imshow(images[i])
            axes[i].set_xlabel(f"t = {t[i]}", fontsize=10)
            
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        else:
            axes[i].axis("off")

    plt.tight_layout()
    fig.suptitle("Generatd Images With Different time Step", fontsize=16)
    plt.subplots_adjust(top=0.9) 
    plt.show()





  def backward_process(self,pred_noise, x_t,t):
    ## shape of pred_nosie (batch_size,channel,h,w)
    ## shape of x_t (batch_size,channel,h,w) where x_t is pure noise , (mean =0,std=1)
    t = t.to(self.device)
    if self.sqrt_alpha_bar[t] > 0:
      reciprocal_aplpha_bar_t = 1/self.sqrt_alpha_bar[t]
    reciprocal_aplpha_bar_t = reciprocal_aplpha_bar_t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # changing the shape from (64,) to (64,1,1,1)

    sqrt_aplpha_bar_t  = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3) # same as uppper line
    x_0 = reciprocal_aplpha_bar_t * (x_t - sqrt_aplpha_bar_t*pred_noise) # shape of x_0 (batch_size,channel,hight,width)

    self.plot_images(x_0)
    return x_0




