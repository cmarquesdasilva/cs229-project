"""Toxicity Classifiers using Roberta-XLM model for multilingual text classification."""
import torch.nn.functional as F
from torch import nn
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    AutoModel,
    AutoTokenizer,
)

def train_model(model, batch, optimizer, device):

    optimizer.zero_grad()
    b_ids = batch['token_ids']
    b_mask = batch['attention_mask']
    b_labels = batch['class_ids']
    b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
    logits = model.forward(b_ids, b_mask) 
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float(), reduction='mean')
    loss.backward()
    optimizer.step()

    return loss.item()

# Main Classes
class ToxicityClassifier(nn.Module):
    """
    Binary Classifier to detect toxicity in multilingual text
    """
    def __init__(self, config):
        super(ToxicityClassifier, self).__init__()

        if config.debug:
            print('Using Bert Tiny')
            self.model_config = BertConfig.from_pretrained('prajjwal1/bert-tiny')
            self.model = BertModel.from_pretrained('prajjwal1/bert-tiny', config=self.model_config)
            self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        else:
            print('Using Roberta-XLM-base')
            self.model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        
        self.classifier = nn.Linear(self.model.config.hidden_size, config.num_classes)
        for name, param in self.model.named_parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
    
    def extract_embeddings(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return cls_output
    
    def tokenizer(self):
        return self.tokenizer
