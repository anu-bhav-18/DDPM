import torch
import torch.nn as nn
import math

class ForwardProcsee(nn.Module):
  def __init__(self,total_time_steps =1000,beta_start= 0.0001,beta_end =0.02,device=None):
    super(ForwardProcsee,self).__init__()
    self.total_time_steps = total_time_steps
    self.beta_start= beta_start
    self.beta_end = beta_end
      
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.betas = torch.linspace(start=self.beta_start,end=self.beta_end,steps=self.total_time_steps, device =self.device)
    self.alphas = 1-self.betas
    alphas_cumprod = torch.cumprod(self.alphas,dim=0)
      
    self.register_buffer('alphas_cumprod', alphas_cumprod)

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
    fig.suptitle("Noised Images With Different time Step", fontsize=16)
    plt.subplots_adjust(top=0.9) 
    plt.show()



  def forward_process(self,images,t,plot_show =False):
    ### images = (batch_size,channel,hight,width)
    ###t= (batch_size,1)
    #print(self.alphas_cumprod.device)
    #print(self.device)

    t = t.to(self.alphas_cumprod.device)
    sqrt_alpha_hat = torch.sqrt(self.alphas_cumprod) #shape of sqrt_alpha_hat(1000,1)
    sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alphas_cumprod)
    sqrt_alpha_hat_t = sqrt_alpha_hat[t]  # shape of sqrt_alpha_hat_t(batch_size,1)
    sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat[t] # shape of sqrt_one_minus_alpha_hat_t(batch_size,1)

    noise = torch.rand_like(images)
    noise = noise.to(self.alphas_cumprod.device)    ## shape as images (batch_size,channel,h,w)
    sqrt_alpha_hat_t =sqrt_alpha_hat_t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # now shape becomes (batch_size,1,1,1)
    sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # same as we do in upper line

    # now we will perform dot product between matrix as element wise product according to paper
    x_t = sqrt_alpha_hat_t * images + sqrt_one_minus_alpha_hat_t * noise
    if plot_show:
        self.plot_images(x_t,t)

    # return the images and noise having shape images(batch_size,channel,h,w) and noise as same (batch_size,channel,h,w)
    return x_t,noise

