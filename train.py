import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser
from utils.preprocess import preprocess_vol
from utils.train_utils import train_predictor
from model.vol import VolTransformer


def main(args):
    save_dir = f'{args.save_dir}_{args.window}days'
    if not os.path.exists(save_dir.split('/')[0]):
        os.makedirs(save_dir.split('/')[0])
    train_config = {"bins": args.bins,
                    "window": args.window,
                    "model_dim": args.model_dim,
                    "n_head": args.n_head,
                    "num_layers": args.num_layers,
                    "initial_lr": args.lr,
                    "gamma": args.gamma}
    
    data = preprocess_vol(args.data_path, args.bins, args.window)
    train_num = int(args.train_ratio * len(data))
    
    train_loader = DataLoader(data[:train_num], batch_size=args.bs, shuffle=True)
    test_bs = min(len(data) - train_num, args.bs)
    test_loader = DataLoader(data[train_num:], batch_size=test_bs, shuffle=True)
    
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = VolTransformer(model_dim=args.model_dim,
                           n_head=args.n_head,
                           num_layers=args.num_layers,
                           bins=args.bins,
                           window=args.window).to(device)

    num_param = 0
    for _, param in model.named_parameters():
        num_param += param.numel()
    print(f'model param size: {num_param}')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    loss_function = nn.MSELoss()
    train_predictor(model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=loss_function,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    bins=args.bins,
                    epoch=args.epoch,
                    device=device,
                    save_dir=save_dir,
                    train_config=train_config)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/all.json')
    parser.add_argument('--save_dir', type=str, default='ckpt/vanilla')
    parser.add_argument('--bins', type=int, default=11)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=0.99)
    parser.add_argument('--model_dim', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.999)
    args = parser.parse_args()
    main(args)