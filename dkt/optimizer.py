from torch.optim import Adam, AdamW
from adamp import AdamP, SGDP

def get_optimizer(model, args):
    if args.optimizer.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    # if args.optimizer == 'adamW':
    if args.optimizer.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # if args.optimizer == 'adamP':
    if args.optimizer == 'adamp':
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=0.01)
    # if args.optimizer == 'SGDP':
    if args.optimizer == 'sgdp':
        optimizer = SGDP(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    
    return optimizer