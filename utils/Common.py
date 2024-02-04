import torch
from utils import Constant

def collate_func(batch, max_seq_len, vocab_source, vocab_target):
    src_token, target_token = zip(*batch)

    batch_size = len(batch)
    src_vocab = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    target_vocab = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    for i in range(batch_size):
        src_vocab[i] = uniformVocab(vocab_source(src_token[i]), max_seq_len, True)
        target_vocab[i] = uniformVocab(vocab_target(target_token[i]), max_seq_len, True)
        
    return src_vocab, target_vocab



def uniformVocab(vocab, max_seq_len, includeStartEnd):
    list = []
    if(includeStartEnd):
        list.append(Constant.SOS_IDX)
    list = list + vocab
    if(includeStartEnd):
        list.append(Constant.EOS_IDX)
    remaining = max_seq_len - len(list)
    for _ in range(remaining):
        list.append(Constant.PAD_IDX)

    tensor = torch.tensor(list, dtype= torch.long)
    return tensor