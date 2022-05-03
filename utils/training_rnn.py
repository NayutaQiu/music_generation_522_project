# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:52:45 2022

@author: nayuta Qiu
"""
import torch
from datetime import datetime, date
def grad_clipping(net, theta):  
    """Clip the gradient."""
    params = [p for p in net.parameters() if p.requires_grad]

    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
def train_epoch_rnn(dataloader, model, optimizer, criterion, device, batch_size, alpha):
    #bug in doc for weight assignment in bcewithlogitloss
    running_loss = 0
    n_obs = 0
    model.train()
    for i, (train_seq, target_seq) in enumerate(dataloader):
        train_seq = train_seq.to(device).float()
        target_seq = target_seq.to(device).float()
        if i==0:
            hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        output, hidden = model(train_seq, hidden)
        #weight = alpha * torch.logical_xor(train_seq[:, :, -1, :].flatten(1,2), target_seq) + 1
        #print(output.shape)
        #print(target_seq.shape)
        #print(output.shape, target_seq.shape)
        loss = criterion(output, target_seq)
        #loss = torch.mean(weight * loss)  #- beta * criterion(output, train_seq[:, :, -1, :].flatten(1,2))
        #loss = criterion(output, target_seq)
        loss.backward()
        hidden = hidden.detach()
        #grad_clipping(model, 1)
        optimizer.step()
        running_loss += loss.item()
        n_obs += 1
    
        
    return running_loss/n_obs 

def test_epoch_rnn(dataloader, model, optimizer, criterion, device,  batch_size,  alpha):
    print("start test")
    model.eval()
    running_loss = 0
    n_obs = 0
    with torch.no_grad():
        for i, (test_seq, target_seq) in enumerate(dataloader):
            test_seq = test_seq.to(device).float()
            target_seq = target_seq.to(device).float()
            if i==0:
                hidden = model.init_hidden(batch_size)
            output, hidden = model(test_seq, hidden)
            #weight = alpha * torch.logical_xor(test_seq[:, :, -1, :].flatten(1,2), target_seq) + 1
            loss = criterion(output, target_seq)
            #loss = torch.mean(weight * loss)  #- beta * criterion(output, test_seq[:, :, -1, :].flatten(1,2))
            running_loss += loss.item()
            n_obs += 1
        return running_loss/n_obs 

def training_rnn(model, optimizer, scheduler, criterion, train_dataloader, 
             test_dataloader, n_epoch, device, batch_size,  alpha, autosave, save_on_epoch):
    train_losses = []
    test_losses = []
    name = "".join([i[0] for i in train_dataloader.dataset.get_track_order()])
    resolution = train_dataloader.dataset.get_resolution()
    seq_len = train_dataloader.dataset.get_seq_len()
    for epoch in range(1, n_epoch + 1):
        print(scheduler.get_last_lr())
        train_epoch_loss = train_epoch_rnn(train_dataloader, model, optimizer, criterion, device,
                                       batch_size = batch_size, alpha=alpha)
        train_losses.append(train_epoch_loss)
        if epoch%save_on_epoch == 0 and autosave:
            torch.save(model.state_dict(), f"{name}_gru_trained_r{resolution}s{seq_len}d{date.today().day}e{epoch}") 
        scheduler.step()
        test_epoch_loss = test_epoch_rnn(test_dataloader, model, optimizer, criterion,  device,
                                     batch_size=batch_size, alpha=alpha)
        test_losses.append(test_epoch_loss)
        
        print('Epoch {}, Train Loss: {}, Test Loss: {}, Time: {}'.format(epoch, train_epoch_loss, test_epoch_loss, datetime.now()))
    if autosave:
        torch.save(model.state_dict(), f"{name}_gru_trained_r{resolution}s{seq_len}d{date.today().day}e{epoch}")
    return train_losses, test_losses