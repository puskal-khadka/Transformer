import torch
from torch import nn 
from src.models.Model import Transformer
from utils.Tokenizer import Tokenizer
from utils import Constant
from utils.Common import uniformVocab
from configparser import ConfigParser
import os

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


model_state_path= "checkpoint/checkpoint_model_state_dict_ep1.pt"
assert os.path.exists(model_state_path), "\nUnable to locate the tained model at the specified path. Make sure you have completed training and are using correct path of the trained model in inference.py"

savedModel = torch.load(model_state_path)  # (Parm: Your own model's state path saved during training). Final trained state will be saved in ./saved dir and checkpoint in ./checkpoint dir.
model_state = savedModel['model_state']
optimizer_sate = savedModel['optimizer_state']
source_vocab = savedModel['source_vocab']
target_vocab = savedModel['target_vocab']

languageConfig = config["Spacy_Language"]
src_language = languageConfig["SRC_LANGUAGE"]
trgt_language = languageConfig["TRGT_LANGUAGE"]

transformer = Transformer(len(source_vocab), len(target_vocab), d_model, d_ff, stack, num_heads, max_seq_len, dropout)
transformer.load_state_dict(model_state)


def translate(src):
    transformer.eval()
    tokenizer = Tokenizer(source_spacy_lang=src_language, target_spacy_lang=trgt_language)
    src_token = tokenizer.getSourceToken(src)
    target_token = [Constant.SOS_STRING_VALUE]
    src = uniformVocab(source_vocab(src_token), max_seq_len, True).unsqueeze(0)

    for i in range(max_seq_len):
        target = uniformVocab(target_vocab(target_token), max_seq_len, False).unsqueeze(0)
        output = transformer(src, target)
        feasible_index = torch.argmax(nn.functional.softmax(output, dim=2)[0][i]).item()
        corresponding_token = target_vocab.lookup_token(feasible_index)
        print(corresponding_token)
        target_token.append(corresponding_token)
        if feasible_index == Constant.EOS_IDX:  #end
            break

    translated = ' '.join(target_token).replace(Constant.SOS_STRING_VALUE, "").replace(Constant.EOS_STRING_VALUE,"").replace(Constant.PAD_STRING__VALUE,"").replace(Constant.UNK_STRING_VALUE,"")  
    return translated

translator = translate("What is your name?")
print(f"Original sentence: What is your name? \nTranslated sentence (nepali): {translator}")