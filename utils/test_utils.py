import json
import torch
import numpy as np
from tqdm import tqdm

def save_model(model, save_dir, train_config):
    torch.save(model.state_dict(), f'{save_dir}.pt')
    with open(f'{save_dir}.json', 'w') as f:
        json.dump(train_config, f)
    
def test_predictor(model, loss_function, dataloader, bins,
                   device, save_dir, train_config, best_test_loss=None,
                   save_ckpt=True, load_ckpt=False):
    if load_ckpt:
        model.load_state_dict(torch.load(f'{save_dir}.pt'))

    model.eval()
    test_loss = 0    
    for idx, batch in tqdm(enumerate(dataloader)):
        data = torch.tensor(batch, dtype=torch.float32).to(device)
        
        out = model(data[:,:-1])
        label = data[:,1:]
        
        loss = loss_function(out.reshape(-1, bins), label.reshape(-1, bins))            
        test_loss += loss.detach().cpu().item()   
        
        if idx == 0:
            print(f'Out: {out[:,-1]}\n Label: {label[:,-1]}')
    
    avg_test_loss = np.sqrt(test_loss / (idx+1))    
    print(f'Test Average Loss: {avg_test_loss}')

    if save_ckpt:
        if best_test_loss is None:
            save_model(model, save_dir, train_config)
        elif avg_test_loss < best_test_loss:
            save_model(model, save_dir, train_config)
            
    return avg_test_loss