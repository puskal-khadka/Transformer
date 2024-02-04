from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from utils import Constant


class CustomDataset(Dataset):

    def __init__(self, lines, src_tokenizer, target_tokenizer):
        super().__init__()
        self.lines = lines
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        src, target = self.lines['english_sent'].iloc[idx],  self.lines['nepali_sent'].iloc[idx]
        src_tokens = self.src_tokenizer(str(src))
        target_tokens = self.target_tokenizer(str(target))
        return [tok.text for tok in src_tokens], [tok.text for tok in target_tokens]  # returning (source token, target token)
    

    def yield_tokens(self, index ):
        for i in range(len(self)):
            yield self[i][index]


    def getVocab(self):
        src_iterator = self.yield_tokens(0)

        vocab_source = build_vocab_from_iterator(
            src_iterator,
            specials=Constant.SPECIAL_SYMBOLS)
        
        
        targ_iterator = self.yield_tokens(1)

        vocab_target = build_vocab_from_iterator(
            targ_iterator,
            specials=Constant.SPECIAL_SYMBOLS)

        vocab_source.set_default_index(Constant.UNK_IDX)
        vocab_target.set_default_index(Constant.UNK_IDX)
        return vocab_source, vocab_target