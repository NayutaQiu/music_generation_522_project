import torch
from datetime import datetime, date
import torch.nn.functional as F
def train_epoch_cnn(dataloader, model, optimizer, criterion, device):
    #bug in doc for weight assignment in bcewithlogitloss
    running_loss = 0
    n_obs = 0
    model.train()
    for train_x, label in dataloader:
        train_x = train_x.float().to(device)
        label = label.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_obs += 1
    
        
    return running_loss/n_obs 

def test_epoch_cnn(dataloader, model, optimizer, criterion, device):
    print("start test")
    model.eval()
    running_loss = 0
    n_obs = 0
    with torch.no_grad():
        for test_x, label in dataloader:
            test_x = test_x.float().to(device)
            label = label.type(torch.LongTensor).to(device)
            output = model(test_x)
            loss = criterion(output, label)
            running_loss += loss.item()
            n_obs += 1
        return running_loss/n_obs 

def training_cnn(model, optimizer, scheduler, criterion, train_dataloader, 
             test_dataloader, n_epoch, device, autosave, save_on_epoch):
    model.train()
    train_losses = []
    test_losses = []
    name = "".join([i[0] for i in train_dataloader.dataset.get_track_order()])
    resolution = train_dataloader.dataset.get_resolution()
    window_len = train_dataloader.dataset.get_window_length()
    for epoch in range(1, n_epoch + 1):
        print(scheduler.get_last_lr())
        train_epoch_loss = train_epoch_cnn(train_dataloader, model, optimizer, criterion, device)
        train_losses.append(train_epoch_loss)
        if epoch%save_on_epoch == 0 and autosave:
            torch.save(model.state_dict(), f"{name}_cnn_trained_r{resolution}w{window_len}d{date.today().day}e{epoch}") 
        scheduler.step()
        test_epoch_loss = test_epoch_cnn(test_dataloader, model, optimizer, criterion,  device)
        test_losses.append(test_epoch_loss)
        
        print('Epoch {}, Train Loss: {}, Test Loss: {}, Time: {}'.format(epoch, train_epoch_loss, test_epoch_loss, datetime.now()))
    if autosave:
        torch.save(model.state_dict(), f"{name}_cnn_trained_r{resolution}w{window_len}d{date.today().day}e{epoch}")
    return train_losses, test_losses

def evaluate_cnn(model, data_loader, device):
    model.eval()
    pred_y = []
    true_y = []
    with torch.no_grad():
        for test_x, label in data_loader:
            test_x = test_x.float().to(device)
            output = F.softmax(model(test_x), dim=1)
            output_label = torch.argmax(output, dim=1, keepdim=False)
            pred_y.append(output_label.cpu().numpy())
            true_y.append(label.cpu().numpy())
        return pred_y, true_y
        