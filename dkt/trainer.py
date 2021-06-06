import os
import json
import torch
import numpy as np

from tqdm.auto import tqdm
from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import *
from lgbm_utils import *

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import wandb

def run(args, train_data, valid_data):
    lgbm_params=args.lgbm.model_params
    
    
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    #lgbm은 학습과 추론을 함께함
    if args.model=='lgbm':
        print("k-fold를 사용하지 않습니다","-"*80)
        model,auc,acc=lgbm_train(args,train_data,valid_data)
        if args.wandb.using:
            wandb.log({"valid_auc":auc, "valid_acc":acc})
        #추론준비
        csv_file_path = os.path.join(args.data_dir, args.test_file_name)
        test_df = pd.read_csv(csv_file_path)#, nrows=100000)

        test_df = make_lgbm_feature(args,test_df)
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        test_df.sort_values(by=['userID','Timestamp'], inplace=True)
        test_df=lgbm_make_test_data(test_df)
        #추론
        lgbm_inference(args,model,test_df)
        return
        
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args,args.model)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        if args.wandb.using:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, f'{args.task_name}.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def run_kfold(args, train_data):
    n_splits = args.n_fold

    ### LGBM runner
    if args.model=='lgbm':
        print("k-fold를 사용합니다","-"*80)
        csv_file_path = os.path.join(args.data_dir, args.file_name)
        train_df = pd.read_csv(csv_file_path)#, nrows=100000)
        
        csv_file_path = os.path.join(args.data_dir, args.test_file_name)
        test_df = pd.read_csv(csv_file_path)#, nrows=100000)
        
        if args.use_test_data:#test의 데이터까지 사용할 경우
            train_df=make_sharing_feature(args)
            
        train_df=make_lgbm_feature(args,train_df)
        test_df=make_lgbm_feature(args,test_df)
        
        delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp','sec_time']
        features=list(set(test_df.columns)-set(delete_feats))

        print(f'사용한 피처는 다음과 같습니다')
        print(features)

        if args.split_by_user: #유저별로 train/valid set을 나눌 때
            y_oof,pred,fi,score,acc=make_lgb_user_oof_prediction(args,train_df, test_df, features, categorical_features='auto', model_params=args.lgbm.model_params, folds=args.n_fold)
            if args.wandb.using:
                wandb.log({"valid_auc":score, "valid_acc":acc})
        else : #skl split라이브러리를 이용하여 유저 구분없이 나눌 때 
            y_oof,pred,fi=make_lgb_oof_prediction(args,train_df, test_df, features, categorical_features='auto', model_params=args.lgbm.model_params, folds=args.n_fold)
        
        
        new_output_path=f'{args.output_dir}{args.task_name}'
        write_path = os.path.join(new_output_path, "output.csv")
        if not os.path.exists(new_output_path):
            os.makedirs(new_output_path)    
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("id,prediction\n")
            for id, p in enumerate(pred):
                w.write('{},{}\n'.format(id,p))

        print(f"lgbm의 예측파일이 {new_output_path}/{args.task_name}.csv 로 저장됐습니다.")

        save_path=f"{args.output_dir}{args.task_name}/feature{len(features)}_config.json"
        json.dump(
            features,
            open(save_path, "w"),
            indent=2,
            ensure_ascii=False,
        )
        return


    if args.use_stratify == True:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True)
    

    target = get_target(train_data)
    val_auc = 0
    val_acc = 0

    oof = np.zeros(train_data.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_data, target)):
        trn_data = train_data[train_idx]
        val_data = train_data[valid_idx]
        
        train_loader, valid_loader = get_loaders(args, trn_data, val_data)

        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10
                
        model = get_model(args,args.model)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_acc = 0
        best_preds = None

        early_stopping_counter = 0
        for epoch in range(args.n_epochs):

            print(f"Start Training: Epoch {epoch + 1}")
            
            ### TRAIN
            train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
            
            ### VALID
            auc, acc, preds , _ = validate(valid_loader, model, args)

            ### TODO: model save or early stopping
            if args.wandb.using:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                    "valid_auc":auc, "valid_acc":acc})

            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_preds = preds
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, 'module') else model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    },
                    (args.model_dir + args.task_name), f'{args.task_name}_{fold+1}fold.pt',
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                    break

            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
            else:
                scheduler.step()
        
        val_auc += best_auc/n_splits
        val_acc += best_acc/n_splits
        oof[valid_idx] = best_preds

    
    print(f'Valid AUC : {val_auc}, Valid ACC : {val_acc} \n')


def train(train_loader, model, optimizer, args):
    model.train()
    print("start training--------------------------")
    total_preds = []
    total_targets = []
    losses = []
    for step, batch in tqdm(enumerate(train_loader)):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[3] # correct


        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[3] # correct


        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets



def inference(args, test_data):
    if args.model=='lgbm':
        return
    
    print("start inference--------------------------")
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in tqdm(enumerate(test_loader)):
        input = process_batch(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)
    
    new_output_path=f'{args.output_dir}{args.task_name}'
    write_path = os.path.join(new_output_path, "output.csv")
    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))



def inference_kfold(args, test_data):
    if args.model=='lgbm':
        return
    
    oof_pred = None

    print("start inference--------------------------")
    for fold in range(args.n_fold):
        print(f'{fold+1} fold inference')
        model = load_model_kfold(args, fold)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)
        
        
        fold_preds = []
        
        for step, batch in tqdm(enumerate(test_loader)):
            input = process_batch(batch, args)
            preds = model(input)

            # predictions
            preds = preds[:,-1]
            
            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()
                
            fold_preds+=list(preds)
        
        fold_pred = np.array(fold_preds)

        if oof_pred is None:
            oof_pred = fold_pred / args.n_fold
        else:
            oof_pred += fold_pred / args.n_fold
        


    new_output_path=f'{args.output_dir}/{args.task_name}'
    write_path = os.path.join(new_output_path, "output.csv")

    if not os.path.exists(new_output_path):
        os.makedirs(new_output_path)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(oof_pred):
            w.write('{},{}\n'.format(id,p))



def get_model(args,model_name:str):
    """
    Load model and move tensors to a given devices.
    """
    if model_name == 'lstm': model = LSTM(args)
    if model_name == 'lstmattn': model = LSTMATTN(args)
    if model_name == 'bert': model = Bert(args)
    if model_name == 'lstmroberta' : model = LSTMRobertaATTN(args)
    if model_name == 'lastquery': model = LastQuery(args)
    if model_name == 'saint': model = Sain(args)
    

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):
    test, question, tag, correct, mask = batch
    

    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    """
    interaction에서 rolling의 이유
    - 이전 time_step에서 푼 문제를 맞췄는지 틀렸는지를 현재 time step의 input으로 넣기 위해서 rolling을 사용한다.
    """

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)
    # print(interaction)
    # exit()
    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)


    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    return (test, question,
            tag, correct, mask,
            interaction, gather_index)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss

def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()



def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))



def load_model(args):
    model_path = os.path.join(args.model_dir, f'{args.task_name}.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args, args.model)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model



def load_model_kfold(args, fold):
    model_path = os.path.join((args.model_dir + args.task_name), f'{args.task_name}_{fold+1}fold.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args, args.model)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model



def get_target(datas):
    targets = []
    for data in datas:
        targets.append(data[-1][-1])

    return np.array(targets)