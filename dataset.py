import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

class ThesisDataset(Dataset):
    def __init__(self, data, test, padding=0, padded_len=500, shuffle=True):
        self.data = data
        self.context_padded_len = padded_len
        self.test = test
        self.shuffle = shuffle
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.padding = padding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = dict(self.data[index])
        if len(data['abstract']) > self.context_padded_len:
            data['abstract'] = data['abstract'][:500]
        return data

    def collate_fn(self, datas):
        batch = {}
        batch['length'] = [data['length'] for data in datas] 
        padded_len = min(self.context_padded_len, max(batch['length']))
        batch['abstract'] = torch.tensor(
            [pad_to_len(data['abstract'], padded_len, self.tokenizer, self.padding)
             for data in datas]
        ).to()
        if not self.test:
            batch['label'] = torch.FloatTensor([data['label'] for data in datas])
        return batch


def pad_to_len(arr, padded_len, tokenizer, padding=0):
    """ Pad `arr` to `padded_len` with padding if `len(arr) < padded_len`.
    If `len(arr) > padded_len`, truncate arr to `padded_len`.
    Example:
        pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
        pad_to_len([1, 2, 3, 4, 5, 6], 5, -1) == [1, 2, 3, 4, 5]
    Args:
        arr (list): List of int.
        padded_len (int)
        padding (int): Integer used to pad.
    """
    # TODO
    length_arr = len(arr)
    new_arr = arr

    if length_arr < padded_len:
        for i in range(padded_len - length_arr):
            new_arr.append(padding)
    else:
        for i in range(length_arr - padded_len):
            del new_arr[-2]
    return new_arr

