from torchtext.data.utils import get_tokenizer
import spacy

class Tokenizer:
    def __init__(self, source_spacy_lang, target_spacy_lang):
        self.tokenizer = {}
        self.tokenizer_source = spacy.blank(source_spacy_lang) 
        self.tokenizer_target = spacy.blank(target_spacy_lang)

    def getSourceToken(self, textDoc):
        src_tokens = self.tokenizer_source(str(textDoc))
        return [tok.text for tok in src_tokens]
    
    def getTargetToken(self, textDoc):
        trg_tokens = self.tokenizer_target(str(textDoc))
        return [tok.text for tok in trg_tokens]
    
    def getSourceAndTargetTokenizer(self):
        return self.tokenizer_source, self.tokenizer_target




