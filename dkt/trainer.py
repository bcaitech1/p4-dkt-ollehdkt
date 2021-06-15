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

from .model import *
from lgbm_utils import *
from .new_model import *

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

import wandb

import torch.cuda.amp

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
            
    model = get_model(args)
#     model = get_model(args,args.model)
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
                
        model = get_model(args)
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

    total_preds = []
    total_targets = []
    losses = []

    for step, batch in enumerate(train_loader):
        gc.collect()
        
        
        if isinstance(model,MyLSTMConvATTN) or isinstance(model,Saint) or isinstance(model, LastQuery_Post) or isinstance(model,LastQuery_Pre)\
            or isinstance(model, TfixupSaint) or isinstance(model,LSTM) or isinstance(model, AutoEncoderLSTMATTN):
            
            input = process_batch_v2(batch, args)
        elif isinstance(model,GeneralizedLSTMConvATTN) or isinstance(model,GeneralizedSaint) or isinstance(model, GeneralizedSaintPlus)\
            or isinstance(model,LastQuery) or isinstance(model,AutoTransformerModel):
            
            input = process_batch_v3(batch,args)
        else:
            input = process_batch(batch,args)
        # print(f"input 텐서 사이즈 : {type(input)}, {len(input)}")
        # with torch.cuda.amp.autocast():
        preds = model(input)
        # targets = input[3] # correct
        targets = input[len(args.cate_cols)]
        # print(f'targets : {targets}')
        # print(f'targets : {targets}')

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
        # input = process_batch(batch, args)
        if isinstance(model,MyLSTMConvATTN) or isinstance(model,Saint) or isinstance(model, LastQuery_Post)\
             or isinstance(model, TfixupSaint) or isinstance(model, AutoEncoderLSTMATTN):
            input = process_batch_v2(batch, args)

        elif isinstance(model,GeneralizedLSTMConvATTN) or isinstance(model,GeneralizedSaint) or isinstance(model, GeneralizedSaintPlus)\
        or isinstance(model,LastQuery) or isinstance(model,AutoTransformerModel):
            # print('process_batch_v3 사용 for validate')
            input = process_batch_v3(batch, args)
        else:
            input = process_batch(batch,args)


        preds = model(input)
        # targets = input[3] # correct
        targets = input[len(args.cate_cols)]


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

    for step, batch in enumerate(test_loader):
        # input = process_batch(batch, args)
        # input = process_batch_test(batch,args)
        if isinstance(model,GeneralizedLSTMConvATTN) or isinstance(model,GeneralizedSaint) or isinstance(model, LastQuery)\
        or isinstance(model,GeneralizedSaintPlus) or isinstance(model,AutoTransformerModel):
            input = process_batch_v3(batch,args)
        else:
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
            # input = process_batch(batch, args)
            input = process_batch_v3(batch,args)
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

# 서빙을 위한 추론
def inference_for_serving(args, test_data):
    
    model = args.model
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch_v3(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    return 100*sum(total_preds)/len(total_preds)


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.work in ['server','test']:
        model = LSTMTest(args)
        model = model.to(args.device)
        return model
    if args.model.lower() == 'lstm': model = LSTMTest(args)
    # if args.model.lower() == 'lstmattn': model = LSTMATTN(args)
    # if args.model.lower() == 'bert': model = Bert(args)
    if args.model.lower() == 'bilstmattn': model = BiLSTMATTN(args)
    if args.model.lower() == 'lstmconvattn' or args.model.lower() == 'lstmrobertaattn' or args.model.lower() == 'lstmattn'\
       or args.model.lower() == 'lstmalbertattn': 
        model = AutoEncoderLSTMATTN(args)
    # if args.model.lower() == 'mylstmconvattn' : model = MyLSTMConvATTN(args)
    if args.model.lower() == 'saint' : model = Saint(args)
    if args.model.lower() == 'lastquery_post': 
        args.lq_padding = 'post'
        model = LastQuery(args)
    if args.model.lower() == 'lastquery_pre' : 
        args.lq_padding = 'pre'
        model = LastQuery(args)
    # if args.model.lower() == 'lastquery_post_test' : model = LastQuery_Post_TEST(args) # 개발중(deprecated)
    if args.model.lower() == 'tfixsaint' : model = TfixupSaint(args) # tfix-up을 적용한 Saint
    if args.model.lower() == 'generalizedlstmconvattn' or args.model.lower() == 'glstmconvattn' : model = GeneralizedLSTMConvATTN(args)
    if args.model.lower() == 'generalizedsaint' or args.model.lower() == 'gsaint': model = GeneralizedSaint(args)
    if args.model.lower() == 'generallizedsaintplus' or args.model.lower() == 'gsaintplus' : model = GeneralizedSaintPlus(args)
    if args.model.lower() in ['bert','gpt2','roberta'] : model = AutoTransformerModel(args)

    model.to(args.device)

    return model

# 배치 전처리(일반화)
def process_batch_v3(batch,args):

    # batch 순서 : (범주형)...,(answerCode), (연속형)..., mask
    
    cate_cols = [batch[i] for i in range(len(args.cate_cols))]
    cont_cols = [batch[i] for i in range(len(args.cate_cols)+1,len(batch)-1)]
 
    correct = batch[len(args.cate_cols)]
    mask = batch[len(batch)-1]
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #  saint의 경우 decoder에 들어가는 input이다
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    # interaction[:, 0] = 0 # set padding index to the first sequence
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)
    
    #  test_id, question_id, tag
    # test = ((test + 1) * mask).to(torch.int64)
    # question = ((question + 1) * mask).to(torch.int64)
    # tag = ((tag + 1) * mask).to(torch.int64)
    for i in range(len(cate_cols)):
        cate_cols[i] = ((cate_cols[i])*mask).to(torch.int64)
        cate_cols[i] = cate_cols[i].to(args.device)

    new_cont = [None for i in range(len(args.cate_cols)+1,len(batch)-1)] # 연속형 컬럼값을 저장하기 위한 list
    
    if args.use_mask :
        new_cont = []
        for i in range(len(args.cate_cols)+1,len(batch)-1):
            cont = batch[i].type(torch.FloatTensor)

            # 롤링 롤링~
            # if use_roll:
            # cont = cont_feature.roll(shifts=1, dims=1)
            # cont[:, 0] = 0 # set padding index to the first sequence
            # cont = (cont_feat * interaction_mask).unsqueeze(-1)

            cont = (cont * mask).unsqueeze(-1)
            # print(cont.shape)
            cont = cont.to(args.device)
            new_cont.append(cont)
    elif args.use_roll: # rolling
        new_cont = []
        for i in range(len(args.cate_cols)+1,len(batch)-1):
            cont = batch[i].type(torch.FloatTensor)

            cont = cont_feature.roll(shifts=1, dims=1)
            cont[:, 0] = 0 # set padding index to the first sequence
            cont = (cont_feat * interaction_mask).unsqueeze(-1)
            cont = cont.to(args.device)
            new_cont.append(cont)
    else:
        # 연속형
        for i in range(len(args.cate_cols)+1,len(batch)-1):
            new_cont[i-(len(args.cate_cols)+1)] = batch[i].to(args.device)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동

    interaction = interaction.to(args.device)
    # m=max(torch.max(interaction),m)
    gather_index = gather_index.to(args.device)

    # 기타 feature를 args의 device에 load
    # for i in range(len(new_batch)):
    #     new_batch[i] = new_batch[i].to(args.device)
    
    ret = []
    ret.extend(cate_cols)
    ret.extend([correct.to(args.device),mask,interaction])
    
    ret.extend(new_cont)
    
    if args.work not in ['test','server']:
        ret.append(gather_index)
    # 반환 :  (범주형),(mask),(interaction),(연속형),(gather_index)
    
    return tuple(ret)

# 배치 전처리 일반화
def process_batch_v2(batch, args):
    # batch : load_data_from 에서 return 시킬 feature(컬럼)
    # test, question,tag,correct, test_level_diff, tag_mean,tag_sum,ans_rate,mask = batch
    # 규칙 : mask는 항상 맨 뒤임
    # print(type(batch))
    # print(batch)
    test = batch[0]
    question = batch[1]
    tag = batch[2]
    correct = batch[3]
    mask = batch[len(batch)-1]
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
    
    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    new_batch = [None for i in range((len(batch)-5))]
    # 기타 features
    for i in range(4,len(batch)-1):
        new_batch[i-4] = ((batch[i]+1)*mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # device memory로 이동
    m=0
    test = test.to(args.device)
    m=max(torch.max(test),m)
    question = question.to(args.device)
    m=max(torch.max(question),m)

    tag = tag.to(args.device)
    m=max(torch.max(tag),m)
    correct = correct.to(args.device)
    m=max(torch.max(correct),m)
    mask = mask.to(args.device)
    m=max(torch.max(mask),m)

    interaction = interaction.to(args.device)
    m=max(torch.max(interaction),m)
    gather_index = gather_index.to(args.device)

    # 기타 feature를 args의 device에 load
    for i in range(len(new_batch)):
        new_batch[i] = new_batch[i].to(args.device)
        m=max(torch.max(new_batch[i]),m)

    ret = [test,question,tag,correct,mask,interaction]
    ret.extend(new_batch)
    ret.append(gather_index)
    # print(f"최댓값 : {m}")
    return tuple(ret)

# 배치 전처리 테스트
def process_batch_test(batch, args):
    # if len(batch)==6:
    #     test, question, tag, correct, test_level_diff, mask = batch
    # else:
    #     test, question, tag, correct, mask = batch
    test, question,tag,correct, test_level_diff, tag_mean,tag_sum,ans_rate,mask = batch
    # test, question, tag, correct, mask = batch # base
    
    # print(type(batch))
    
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
    # print(interaction)
    # exit()
    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)
    
    test_level_diff = ((test_level_diff+1) * mask).to(torch.int64)

    tag_mean = ((tag_mean+1) * mask).to(torch.int64)
    tag_sum = ((tag_sum+1) * mask).to(torch.int64)
    ans_rate = ((ans_rate+1) * mask).to(torch.int64)

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

    test_level_diff = test_level_diff.to(args.device)
    tag_mean = tag_mean.to(args.device)
    tag_sum = tag_sum.to(args.device)
    ans_rate = ans_rate.to(args.device)

    return (test, question,
    tag, correct, mask, interaction, test_level_diff,tag_mean,tag_sum,ans_rate, gather_index)

# 배치 전처리(기본 feature만 쓸 때, baseline)
def process_batch(batch, args):
    if len(batch)==6:
        test, question, tag, correct, test_level_diff, mask = batch
    else:
        test, question, tag, correct, mask = batch
    # test, question, tag, correct, mask = batch # base
    # print(type(batch))
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
    if len(batch)==6: # train시
        test_level_diff = ((test_level_diff+1) * mask).to(torch.int64)


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

    # dev
    if len(batch)!=6:
        
        return (test, question,
            tag, correct, mask,
            interaction, gather_index)
    test_level_diff = test_level_diff.to(args.device)
    # return (test, question,
    #         tag, correct, mask,
    #         interaction, gather_index) # base
    return (test, question,
    tag, correct, mask, interaction, test_level_diff, gather_index)


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

def get_target(datas):
    targets = []
    for data in datas:
        targets.append(data[-1][-1])

    return np.array(targets)

def load_model(args):
    if args.work in ['test','server']:
        print('load model for serving...')
        model_path = os.path.join(args.model_dir,f'model-lstm.pt')
    else:
        print('load model for train and inf')
        model_path = os.path.join(args.model_dir, f'{args.task_name}.pt')
    # model_path = os.path.join(args.model_dir,f'model-lstm.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model
