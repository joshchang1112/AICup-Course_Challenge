import os
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from metrics import Recall
from preprocess import preprocess
from dataset import ThesisDataset
from utils import set_seed
from sklearn.model_selection import KFold
from transformers import RobertaForSequenceClassification

def run_iter(batch, model, device, training):
    context, context_lens = batch['abstract'].to(device), batch['length']
    batch_size = context.size()[0]
    max_context_len = context.size()[1]
    padding_mask = []
    for i in range(batch_size):
        if context_lens[i] < max_context_len:
            tmp = [1] * context_lens[i] + [0] * (max_context_len - context_lens[i])
        else:
            tmp = [1] * max_context_len
        padding_mask.append(tmp)
    padding_mask = torch.Tensor(padding_mask).to(device)
    
    if training:
        prob = F.sigmoid(model(context, attention_mask=padding_mask)[0])
    else:
        with torch.no_grad():
            prob = F.sigmoid(model(context, attention_mask=padding_mask)[0])
    return prob

def training(args, train_loader, valid_loader, model, device, split):
    """Training Procedure"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()
    metric = Recall()
    total_iter = 0
    best_valid_f1 = 0
     
    for epoch in range(args.epochs):
        train_trange = tqdm(enumerate(train_loader), total=len(train_loader), desc='training')
        train_loss = 0
        for i, batch in train_trange:
            answer = batch['label'].to(device)
            prob = run_iter(batch, model, device, training=True)
            loss = criterion(prob, answer)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iter += 1
            metric.update(prob, answer)
            train_trange.set_postfix(
                loss=train_loss/(i+1),
                **{metric.name: metric.print_score()})

            if total_iter % args.eval_steps == 0:
                valid_f1 = validation(valid_loader, model, device)
                if valid_f1 > best_valid_f1:
                    best_valid_f1 = valid_f1
                    torch.save(model, os.path.join(args.model_dir, 'fine_tuned_roberta_{}.pkl'.format(split)))

def validation(dataloader, model, device):
    metric = Recall()
    criterion = torch.nn.BCELoss()
    valid_trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='validation')
    model.eval()
    valid_loss = 0
    for i, batch in valid_trange:
        prob = run_iter(batch, model, device, training=False)
        answer = batch['label'].to(device)
        loss = criterion(prob, answer)
        valid_loss += loss.item()
        metric.update(prob, answer)
        valid_trange.set_postfix(
            loss=valid_loss/(i+1),
            **{metric.name: metric.print_score()})
    return metric.get_f1()

def testing(args, dataloader, model, device, split):
    header = ['Id', 'THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='testing')
    model.eval()
    for i, batch in trange:
        prob = run_iter(batch, model, device, training=False)
        predict = torch.where(prob > 0.4, torch.ones_like(prob), torch.zeros_like(prob)).cpu().detach().numpy()
        index = np.expand_dims(np.arange(i*args.batch_size+1, (i+1)*args.batch_size+1), axis=1)
        predict = np.concatenate((index, predict), axis=1)
        if i == 0:
            total_predict = predict
        else:
            total_predict = np.concatenate((total_predict, predict), axis=0)
    
    total_predict = total_predict.astype(int) 
    pd.DataFrame(total_predict).to_csv(os.path.join(args.result_dir, 'predict-{}.csv'.format(split)), 
                 index=False, header=header)
    
def main():    
    parser = argparse.ArgumentParser(description='Thesis Classification')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--num_split', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--split', type=int, default=-1,
                        help='-1 means all split, 0~args.num_split-1 are availabe numbers.')
    parser.add_argument('--model_dir', default='models', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--result_dir', default='results', type=str,
                        help='Directory to the result csv.')
    args = parser.parse_args()

    if args.device == -1:
        device = torch.device('cpu')
        print("Use CPU to train!")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.device))
            print("Use GPU {} to train!".format(args.device))
        else:
            print("Cuda is not available! Please check your cuda version. Use CPU to train now!")
            device = torch.device('cpu')
	    
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
        print('Create model folder: {}'.format(args.model_dir))
    else:
        print('{} exists.'.format(args.model_dir))
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
        print('Create result folder: {}'.format(args.result_dir))
    else:
        print('{} exists.'.format(args.result_dir))

    set_seed(args.seed)
    print('Loading dataset...')
    train_data = pd.read_csv('data/trainset.csv')
    test_data = pd.read_csv('data/testset.csv')
   
     
    print('Preprocess dataset...')
    train, test = preprocess(train_data, test_data)
    kf = KFold(n_splits=args.num_split)
    
    for i, (train_index, valid_index) in enumerate(kf.split(train)):
        train_set = [train[idx] for idx in train_index]
        valid_set = [train[idx] for idx in valid_index]

        print('Split {}: Construct dataset and dataloader...'.format(i+1))
        train_dataset = ThesisDataset(train_set, test=False, padded_len=args.max_seq_length)
        valid_dataset = ThesisDataset(valid_set, test=False, padded_len=args.max_seq_length)
        test_dataset = ThesisDataset(test, test=True, padded_len=args.max_seq_length)
    
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4).to(device)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
            shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, 
            shuffle=False, collate_fn=valid_dataset.collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
            shuffle=False, collate_fn=test_dataset.collate_fn)    
        print('Split {}: Start Training!'.format(i+1))
        training(args, train_loader, valid_loader, model, device, split=i+1)
        print('Split {}: Start Predicting!'.format(i+1))
        model = torch.load(os.path.join(args.model_dir, 'fine_tuned_roberta_{}.pkl'.format(i+1)))
        testing(args, test_loader, model, device, split=i+1)
        print('Split {}: Finished!'.format(i+1))
        

if __name__ == '__main__':
    main() 
    

