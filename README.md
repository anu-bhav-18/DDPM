# DDPM Implementation from Scratch (PyTorch)

A clean, modular implementation of **Denoising Diffusion Probabilistic Models (DDPM)** based on the original paper:

**“Denoising Diffusion Probabilistic Models” — Ho et al., 2020**

This project focuses on building the **core concepts from scratch**, including the forward diffusion process, U-Net architecture, and training pipeline, without relying on high-level diffusion libraries.

---

##  Project Overview

This repository implements:

* Forward diffusion (noise scheduling)
* Reverse denoising process (learned via neural network)
* U-Net architecture with:

  * Residual blocks
  * Group Normalization
  * Self / Multi-head attention
  * Time-step embedding injection
* Training loop using noise prediction objective

The goal is to **understand DDPM at a mathematical and implementation level**, not just use prebuilt APIs.

---

##  Key Concepts Implemented

### 1. Forward Diffusion Process

Noise is gradually added to an image over time steps:

![equation](https://latex.codecogs.com/png.image?\dpi{120}x_t%20=%20\sqrt{\hat{\alpha}_t}\cdot x_0%20+%20\sqrt{1-\hat{\alpha}_t}\cdot\epsilon)

* Linear beta schedule
* Precomputed cumulative product of alphas
* Efficient tensor indexing for batch timesteps

---

### 2. Reverse Process (Model Learning)

The model learns to predict noise:

![epsilon_theta](https://latex.codecogs.com/png.image?\dpi{120}\epsilon_{\theta}(x_t,%20t))

Training objective:

* Mean Squared Error (MSE) between predicted noise and true noise

---

### 3. Time Embedding

* Sinusoidal positional encoding for timesteps
* MLP projection (SiLU activation)
* Injected into each residual block

---

### 4. U-Net Architecture

Custom-built U-Net with:

* Encoder–Decoder structure
* Skip connections
* Downsampling / Upsampling blocks
* Attention layers at deeper resolutions
* Residual blocks with time conditioning
---

##  Training Details

* Loss: Mean Squared Error (noise prediction)
* Optimizer: Adam
* Time steps: 1000
* Image normalization: [-1, 1]
* Batch processing with timestep sampling per image

---

##  Visualization

* Intermediate noisy images plotted at different timesteps
* Helps verify correctness of forward diffusion
* Grid-based visualization for batch samples

---

##  Kaggle Notebook

This project was developed and executed in a **Kaggle Notebook**, leveraging:

* Free GPU acceleration
* Easy dataset integration
* Version-controlled experimentation

👉 All training, debugging, and visualization were performed within the Kaggle environment.

* open:- https://www.kaggle.com/code/anubhvkumar47/ddpm-implementation/edit


---

##  Learning Outcomes

* Deep understanding of diffusion probabilistic models
* Hands-on experience with U-Net design in generative models
* Practical implementation of time-conditioned neural networks
* Debugging tensor shapes, masking, and GPU execution

---

##  Future Improvements

* Add classifier-free guidance
* Implement DDIM sampling
* Improve sampling speed
* Integrate latent diffusion (Stable Diffusion style)
* Switch to DistributedDataParallel (DDP)

---

##  Conclusion

This implementation demonstrates that DDPM can be built from first principles using PyTorch by carefully combining:

* probabilistic modeling
* neural network design
* and efficient tensor operations

---

##  Reference

* Ho, Jonathan, Ajay Jain, and Pieter Abbeel.
  *Denoising Diffusion Probabilistic Models*, 2020.

---

##  Notes

This project is intended for **educational and research purposes**, focusing on clarity, modularity, and conceptual correctness rather than production optimization.
