
# Copyright 2026 Anubhav Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
def train(model,loader,loss_fn,optimizer,noise_adder,device,epochs):
    epoch_loss =[]
    for  i in range(epochs):
        model.train()
        total_loss =0
        for b_idx,(images,label) in enumerate(loader):
            images = images.to(device)
            label = label.to(device)
            t = torch.randint(0,999,(images.size(0),))
            x_t,true_noise = noise_adder.forward_process(images,t)
            pred_noise = model(x_t,t)

            loss = loss_fn(pred_noise,true_noise)
            total_loss +=loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss.append(total_loss/len(loader))

        
        print(f"Epoch:-{i},Loss:-{total_loss}")
            
            
            
    model = UNet(in_channels=3, base_channels=128, time_dim=512)


if torch.cuda.device_count()>1:
    print("using multiple gpu:-{torch.cuda.device_count()}")
    model = torch.nn.DataParallel(model)

loss = torch.nn.MSELoss()
otimizer = torch.optim.Adam(model.parameters(),lr=0.003)
fw = ForwardProcsee()

loss = train(model=model,loader=loader,loss_fn=loss,optimizer=otimizer,noise_adder=fw,device=device,epochs=1)
