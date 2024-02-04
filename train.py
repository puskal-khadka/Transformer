import torch
from torch import nn 
import torch.optim as optim
from functools import partial
from sklearn.model_selection import train_test_split
import pandas as pd
from configparser import ConfigParser
from tqdm import tqdm

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from utils import Tokenizer, collate_func
from src.models.Model import Transformer
from src.datasets import CustomDataset

config = ConfigParser()
config.read("Config.ini")

transformerConfig = config["Transformer"]

d_model = int(transformerConfig["d_model"])
num_heads = int(transformerConfig["head_count"])
stack =int ( transformerConfig["stack"])
d_ff = int (transformerConfig["d_ff"])
dropout = float (transformerConfig["dropout"])
max_seq_len = int (transformerConfig["max_seq"])
learning_rate = float (transformerConfig["learning_rate"])
batch_size = int (transformerConfig["batch_size"])
epoch_count = int(transformerConfig['epoch'])

languageConfig = config["Spacy_Language"]
src_language = languageConfig["SRC_LANGUAGE"]
trgt_language = languageConfig["TRGT_LANGUAGE"]


def train(model:Transformer, datasets:CustomDataset, sourceVocab, targetVocab):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_func, max_seq_len= max_seq_len,vocab_source = source_vocab, vocab_target=target_vocab ))
    for epoch in range(epoch_count):
        with tqdm(enumerate(train_dataloader), unit="batch", total=len(train_dataloader)) as tqm:
            for i , ( src_data, tgt_data ) in tqm :
                optimizer.zero_grad()
                output = model(src_data, tgt_data[:, :-1])
                loss = criterion(output.contiguous().view(-1, len(targetVocab)), tgt_data[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                tqm.set_postfix(loss = loss.item())
                tqm.set_description(f" Epoch {epoch+1}/{epoch_count}")
                saveModel(transformer, optimizer, False, epoch, sourceVocab, targetVocab,  f"checkpoint_model_state_dict_ep{epoch+1}")  #  checkpoint

    saveModel(transformer, optimizer, True, epoch_count, sourceVocab, targetVocab, f"trained_model_state_dict")  # trained model
    print("Training Completed")
  
  

def saveModel (model, optimizer, isTrainingFinish, epoch, source_vocab, target_vocab, dictName) :
    save_dict = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "source_vocab": source_vocab,
            "target_vocab": target_vocab
        }
    rootPath = "saved" if isTrainingFinish else "checkpoint"
    torch.save(save_dict ,  f"{rootPath}/{dictName}.pt" )


def loadData():
    df =  pd.read_excel("data/english-nepali.xlsx")
    df = df [ (df['english_sent'].str.len()<=max_seq_len ) & (df['nepali_sent'].str.len()<=max_seq_len) ] #clean
    return df


if __name__ == '__main__' :
    df = loadData()
    tokenizer = Tokenizer(source_spacy_lang=src_language, target_spacy_lang=trgt_language)
    source_tokenizer, target_tokenizer = tokenizer.getSourceAndTargetTokenizer()

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    train_ds = CustomDataset(train_df, source_tokenizer,target_tokenizer)
    val_ds = CustomDataset(test_df, source_tokenizer,target_tokenizer) 
    print("......")

    source_vocab, target_vocab = train_ds.getVocab()
    transformer = Transformer(len(source_vocab), len(target_vocab), d_model, d_ff, stack, num_heads, max_seq_len, dropout)
    train(transformer, train_ds, source_vocab, target_vocab)


    