"""
Created on Sun Apr 24 17:11:27 2022

@author: nayuta Qiu
"""
import torch
from datetime import date, datetime
def train_vae_epoch(dataloader, vae, optimizer, device, nsample):
    running_loss = 0
    n_obs = 0
    vae.train()
    for train_seq, target_seq in dataloader:
        train_seq = train_seq.to(device).float()
        target_seq = target_seq.to(device).float()
        optimizer.zero_grad()
        loss = -vae.elbo(train_seq)
        #print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_obs += train_seq.size()[0]
    return running_loss/n_obs * 100

def test_vae_epoch(dataloader, vae,device, nsample):
    running_loss = 0
    n_obs = 0
    vae.eval()
    with torch.no_grad():
        for test_seq, target_seq in dataloader:
            test_seq = test_seq.to(device).float()
            target_seq = target_seq.to(device).float()
            loss = -vae.elbo(test_seq)
            running_loss += loss.item()
            n_obs += test_seq.size()[0]
    return running_loss/n_obs * 100

def training_vae(model, optimizer, scheduler, train_loader, test_loader, n_epoch, nsample, device, autosave, save_on_epoch):
    train_losses = []
    test_losses = []
    for epoch in range(1, n_epoch+1):
        train_epoch_loss = train_vae_epoch(train_loader, model, optimizer, device, nsample)
        name = "".join([i[0] for i in train_loader.dataset.get_track_order()])
        resolution = train_loader.dataset.get_resolution()
        seq_len = train_loader.dataset.get_seq_len()
        if epoch%save_on_epoch == 0 and autosave:
            torch.save(model.state_dict(), f"{name}_vae_trained_r{resolution}s{seq_len}d{date.today().day}e{epoch}") 
        train_losses.append(train_epoch_loss)
        scheduler.step()
        test_epoch_loss = test_vae_epoch(test_loader, model, device, nsample)
        test_losses.append(test_epoch_loss)
        print('Epoch {}, Train Loss: {}, Test Loss: {}, Time: {}'.format(epoch,
                                                                          train_epoch_loss, test_epoch_loss, datetime.now()))
    return train_losses, test_losses

