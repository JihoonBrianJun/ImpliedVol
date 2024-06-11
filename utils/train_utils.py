import torch
import numpy as np
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
from .test_utils import test_predictor

def train_predictor(model, optimizer, scheduler, loss_function,
                    train_loader, test_loader, bins,
                    epoch, device, save_dir, train_config):
    
    best_test_loss = np.inf
    for epoch in tqdm(range(epoch)):
        if epoch % 10 == 0 and epoch != 0:
            print(f'Epoch {epoch} Average Loss: {epoch_avg_loss}')
            test_loss = test_predictor(model, loss_function, test_loader, bins,
                                       device, save_dir, train_config, best_test_loss)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                
        model.train()
        epoch_loss = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            data = batch.to(device)
            
            out = model(data[:,:-1])
            label = data[:,1:]
            loss = loss_function(out.reshape(-1, bins), label.reshape(-1, bins))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.detach().cpu().item()     
        
        epoch_avg_loss = np.sqrt(epoch_loss/(idx+1))
        scheduler.step()
    
    test_predictor(model, loss_function, test_loader, bins,
                   device, save_dir, train_config, save_ckpt=False, load_ckpt=True)
