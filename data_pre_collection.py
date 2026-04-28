# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from torch.utils.data import DataLoader
transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],
                             std=[0.5,0.5,0.5])
         ])
dataset = datasets.ImageFolder(root="/kaggle/input/datasets/vishalsubbiah/pokemon-images-and-types",transform = transform)
loader = DataLoader(dataset=dataset,num_workers=2,batch_size=64,shuffle=True,pin_memory=True,persistent_workers=True)
loader.batch_size

images , label = next(iter(loader))


plt.imshow(images[0][0])


print(f"TOTAL BATCH IN DATASET:-{len(loader)}")
print(f"SHAPE OF ONE BATCH:-{images.shape}")
print(f"SHAPE OF LABEL :- {label.shape}")


fw = ForwardProcsee()

fw.forward_process(images,torch.randint(0,999,(images.size(0),)),plot_show =True)