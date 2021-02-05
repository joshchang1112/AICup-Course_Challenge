from tqdm import tqdm
from transformers import RobertaTokenizer

def label_to_onehot(labels):
    label_dict = {'THEORETICAL': 0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}
    label = []
    onehot = [0, 0, 0, 0]
    for s in labels.split():
        onehot[label_dict[s]] = 1
    return onehot

def preprocess_sample(data, tokenizer):
    processed = {}
    abstract = data['Abstract'].replace('$$$', ' ')
    processed['abstract'] = tokenizer.tokenize(text=data['Title'] + abstract) 
    processed['abstract'] = tokenizer.convert_tokens_to_ids(processed['abstract'])
    processed['length'] = len(processed['abstract'])
    #processed['length'].append(len(processed['abstract']))
        
    if 'Classifications' in data:
        processed['label'] = label_to_onehot(data['Classifications'])
        
    length = len(processed['abstract'])
    return processed, length

def preprocess(train_data, test_data):
    train, test = [], []
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    for i in tqdm(range(len(train_data))):
        sample, length = preprocess_sample(train_data.iloc[i, :], tokenizer)
        train.append(sample)
    
    for i in tqdm(range(len(test_data))):
        sample, length = preprocess_sample(test_data.iloc[i, :], tokenizer)
        test.append(sample)

    return train, test

