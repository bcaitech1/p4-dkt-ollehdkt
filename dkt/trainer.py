import os
from numpy.lib.arraysetops import isin
import torch
import numpy as np
import json
import gc
from tqdm.auto import tqdm

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
import wandb

from .models_architecture import *
from lgbm_utils import *
# from .new_model import Bert,LSTMATTN

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import wandb

def run(args, train_data, valid_data):
    lgbm_params=args.lgbm.model_params
    print(f'{args.model}모델을 사용합니다')
    print('-'*80)
    
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    #lgbm은 학습과 추론을 함께함
    if args.model=='lgbm':
        print("k-fold를 사용하지 않습니다","-"*80)
        model,auc,acc, precision, recall, f1=lgbm_train(args,train_data,valid_data)
        if args.wandb.using:
            wandb.log({"valid_auc": train_auc, "valid_acc":train_acc,"valid_precision":precision, "valid_recall":recall, "valid_f1": f1})
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
            
    model = get_model(args)
#     model = get_model(args,args.model)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    best_acc = -1
    best_precision=-1
    best_recall=-1
    best_f1=-1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_precision,train_recall,train_f1, train_loss = train(train_loader, model, optimizer, args)
    
        ### VALID
        auc, acc,precision,recall,f1, preds , _ = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        if args.wandb.using:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                "train_precision": train_precision,"train_recall":train_recall,"train_f1":train_f1,
                  "valid_auc":auc, "valid_acc":acc,"valid_precision":precision,"valid_recall":recall,"valid_f1":f1})
        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            best_precision=precision
            best_recall=recall
            best_f1=f1
            best_preds=preds
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
    
    print(f"best AUC : {best_auc:.5f}, accuracy : {best_acc:.5f}, precision : {best_precision:.5f}, recall : {best_recall:.5f}, f1 : {best_f1:.5f}")


def run_kfold(args, train_data):
    n_splits = args.n_fold
    print("k-fold를 사용합니다","-"*80)
    ### LGBM runner
    if args.model=='lgbm':
        
        csv_file_path = os.path.join(args.data_dir, args.file_name)
        train_df = pd.read_csv(csv_file_path)#, nrows=100000)

        csv_file_path = os.path.join(args.data_dir, args.test_file_name)
        test_df = pd.read_csv(csv_file_path)#, nrows=100000)
 
        if args.use_test_data:#test의 데이터까지 사용할 경우
            train_df=make_sharing_feature(args)
            
        train_df=make_lgbm_feature(args,train_df)

        if args.use_distance:
            test_df['distance']=np.load('/opt/ml/np_test_tag_distance_arr.npy')
        
        test_df=make_lgbm_feature(args,test_df)
        
        delete_feats=['userID','assessmentItemID','testId','answerCode','Timestamp','sec_time']
        features=list(set(test_df.columns)-set(delete_feats))

        print(f'사용한 피처는 다음과 같습니다')
        print(features)

        if args.split_by_user: #유저별로 train/valid set을 나눌 때
            y_oof,pred,fi,score,acc, precision, recall, f1=make_lgb_user_oof_prediction(args,train_df, test_df, features, categorical_features='auto', model_params=args.lgbm.model_params, folds=args.n_fold)
            
        else : #skl split라이브러리를 이용하여 유저 구분없이 나눌 때 
            y_oof,pred,fi, score,acc, precision, recall, f1=make_lgb_oof_prediction(args,train_df, test_df, features, categorical_features='auto', model_params=args.lgbm.model_params, folds=args.n_fold)
        
        if args.wandb.using:
            wandb.log({"valid_auc":score, "valid_acc":acc, "valid_precision": precision, "valid_recall": recall,"valid_f1": f1})
        
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
    val_precision=0
    val_recall=0
    val_f1=0
    oof = np.zeros(train_data.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_data, target)):
        print(f'{fold}fold를 수행합니다')
        trn_data = train_data[train_idx]
        val_data = train_data[valid_idx]
        
        train_loader, valid_loader = get_loaders(args, trn_data, val_data)

        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10
                
        model = get_model(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_acc = 0
        best_precision=-1
        best_recall=-1
        best_f1=-1
        best_preds = None

        early_stopping_counter = 0
        for epoch in range(args.n_epochs):

            print(f"Start Training: Epoch {epoch + 1}")
            
            ### TRAIN
            train_auc, train_acc, train_precision,train_recall,train_f1, train_loss = train(train_loader, model, optimizer, args)
        
            ### VALID
            auc, acc,precision,recall,f1, preds , _ = validate(valid_loader, model, args)

            ### TODO: model save or early stopping
            if args.wandb.using:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                "train_precision": train_precision,"train_recall":train_recall,"train_f1":train_f1,
                  "valid_auc":auc, "valid_acc":acc,"valid_precision":precision,"valid_recall":recall,"valid_f1":f1})
        
            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_precision=precision
                best_recall=recall
                best_f1=f1
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
        val_precision += best_precision/n_splits
        val_recall += best_recall/n_splits
        val_f1 += best_f1/n_splits

        oof[valid_idx] = best_preds

    
    print(f"Valid AUC : {val_auc:.5f}, accuracy : {val_acc:.5f}, precision : {val_precision:.5f}, recall : {val_recall:.5f}, f1 : {val_f1:.5f}")


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    
    for step, batch in enumerate(train_loader):
        gc.collect()
        if isinstance(model,Saint) or isinstance(model, LastQuery_Post) or isinstance(model,LastQuery_Pre)\
             or isinstance(model, TfixupSaint) or isinstance(model,LSTM) or isinstance(model, LSTMATTN):
            input = process_batch_v2(batch, args)
        else:
            input = process_batch_v2(batch,args)
        # print(f"input 텐서 사이즈 : {type(input)}, {len(input)}")
        preds = model(input)
        targets = input[-1] # correct


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
    auc, acc ,precision,recall,f1 = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc:.5f} ACC : {acc:.5f} precision : {precision:.5f} recall : {recall:.5f} f1 : {f1:.5f}')
    return auc, acc ,precision,recall,f1, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        # input = process_batch(batch, args)
        if isinstance(model,Saint) or isinstance(model, LastQuery_Post) or isinstance(model,LastQuery_Pre)\
             or isinstance(model, TfixupSaint) or isinstance(model, AutoEncoderLSTMATTN):
            input = process_batch_v2(batch, args)
        else:
            input = process_batch_v2(batch,args)


        preds = model(input)
        targets = input[-1] # correct


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
    auc, acc ,precision,recall,f1 = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc:.5f} ACC : {acc:.5f} precision : {precision:.5f} recall : {recall:.5f} f1 : {f1:.5f}')

    return auc, acc ,precision,recall,f1, total_preds, total_targets



def inference(args, test_data):
    if args.model=='lgbm':
        return
    
    print("start inference--------------------------")
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    print("test_loader에 대해")
    print(len(test_loader))
    print(test_loader)
    
    total_preds = []

    for step, batch in enumerate(test_loader):
        input = process_batch_v2(batch,args)

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
    print("정답의 개수 :",len(total_preds))
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
            input = process_batch_v2(batch, args)
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


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model.lower() == 'lstm': model = LSTM(args)
    # if args.model.lower() == 'lstmattn': model = LSTMATTN(args)
    if args.model.lower() == 'bert': model = Bert(args)
    if args.model.lower() == 'bilstmattn': model = BiLSTMATTN(args)
    if args.model.lower() == 'lstmattn': model = LSTMATTN(args)
    # if args.model.lower() == 'lstmconvattn' or args.model.lower() == 'lstmrobertaattn' or args.model.lower() == 'lstmattn'\
    #    or args.model.lower() == 'lstmalbertattn': 
    #     model = AutoEncoderLSTMATTN(args)
    if args.model.lower() == 'mylstmconvattn' : model = MyLSTMConvATTN(args)
    if args.model.lower() == 'saint' : model = Saint(args)
    if args.model.lower() == 'lastquery_post': model = LastQuery_Post(args)
    if args.model.lower() == 'lastquery_pre' : model = LastQuery_Pre(args)
    if args.model.lower() == 'lastquery_post_test' : model = LastQuery_Post_TEST(args) # 개발중(deprecated)
    if args.model.lower() == 'tfixsaint' : model = TfixupSaint(args) # tfix-up을 적용한 Saint

    model.to(args.device)

    return model

# 배치 전처리 일반화
def process_batch_v2(batch, args):

    cate_cols=batch[:len(args.cate_feats)-1] #userID 빼고
    # print("cate_cols",cate_cols)
    cont_cols=batch[len(args.cate_feats)-1:-2] #answercode, mask 빼고
    # print("cont_cols",cont_cols)
    correct = batch[-2]
    mask = batch[-1]
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)
    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    # interaction[:, 0] = 0 # set padding index to the first sequence
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)
    

    # cate features masking
    for i in range(len(args.cate_feats)-1):
        batch[i] = ((batch[i]+1)*mask).to(torch.int64)
    # cont features masking
    for j in range(len(args.cate_feats)-1,len(batch)-2):
        batch[j] = batch[j].type(torch.FloatTensor) 
        batch[j] = ((batch[j]+1)*mask).to(torch.float32)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1
    
    # feature들 device에 load
    for i in range(len(batch)):
        batch[i] = batch[i].to(args.device)

    correct = correct.to(args.device)
    mask=mask.to(args.device)
    interaction=interaction.to(args.device)
    gather_index = gather_index.to(args.device)
    #userID, answerCode 제거, answer는 이미 get_target에서 갖고갔음
    ret = batch[:len(args.cate_feats)-1]+batch[len(args.cate_feats)-1:len(batch)-2]
    ret.append(interaction)
    ret.append(mask)
    ret.append(gather_index)
    ret.append(correct)
    # print("모델로 넘기는 ret 출력",ret)
    # print(f"최댓값 : {m}")
    return tuple(ret) #tuple(cate + cont + interaction + mask + gather_index + correct)

# 배치 전처리(기본 feature만 쓸 때, baseline)
def process_batch(batch, args):
    # print("배치",batch)
    # print('process batch의 사이즈',len(batch))
    test, question, tag, solve_time,correct, mask = batch

    # print("시간",solve_time)
    # test, question, tag, correct, mask = batch # base
    # print(type(batch))
    # change to float
    solve_time = solve_time.type(torch.FloatTensor)
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
    solve_time=((solve_time + 1) * mask).to(torch.float32)
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
    solve_time=solve_time.to(args.device)
    # dev
    # return (test, question,
    #         tag, correct, mask,
    #         interaction, gather_index) # base
    return (test, question,
    tag, correct, mask, interaction, solve_time, gather_index)

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

def load_model_kfold(args, fold):
    model_path = os.path.join((args.model_dir + args.task_name), f'{args.task_name}_{fold+1}fold.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model

def get_target(datas): #처리하기 전에 get_data_from_file 에서 맨마지막 answer이므로 바뀔일 없음
    targets = []
    for data in datas:
        targets.append(data[-1][-1])

    return np.array(targets)

def load_model(args):
    model_path = os.path.join(args.model_dir, f'{args.task_name}.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model
