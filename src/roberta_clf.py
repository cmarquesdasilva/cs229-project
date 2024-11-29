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
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn.init as init


def initialize_weights(module):
    """
    Initializes weights of the given module.
    """
    if isinstance(module, nn.Linear):
        # Use Xavier Initialization for Linear layers
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm weights and biases
        init.ones_(module.weight)
        init.zeros_(module.bias)


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

        if config.model == 'bert-tiny':
            print('Using Bert Tiny')
            self.model_config = BertConfig.from_pretrained('prajjwal1/bert-tiny')
            self.model = BertModel.from_pretrained('prajjwal1/bert-tiny', config=self.model_config)
            self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
        elif config.model == 'roberta-xlm-base':
            print('Using Roberta-XLM-base')
            self.model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        else:
            raise ValueError('Model not supported')
        
        # Add dropout layers
        self.toxic_dropout = nn.Dropout(p=config.dropout_rate)

        # Add a classifier head
        self.toxic_cls = nn.Linear(self.model.config.hidden_size, config.num_classes)

        if config.option == 'lora':
            lora_config = LoraConfig(r=config.r, lora_alpha=1, lora_dropout=0.1,
            target_modules=['self.query', 'self.key', 'self.value'])
            self.model = get_peft_model(self.model, lora_config)
            
        for name, param in self.model.named_parameters():
            if config.option == 'pretrain':
              param.requires_grad = False
            elif config.option == 'lora':
              if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = True

        # Initialize new layers
        self.apply(initialize_weights)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        x = last_hidden_state[:, 0, :]
        x = self.toxic_dropout(x)
        logits = self.toxic_cls(x)
        return logits
    
    def extract_embeddings(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return cls_output

    def merge_lora(self):
      """Merge LoRA weights into the base model and unload LoRA layers."""
      if hasattr(self.model, "merge_and_unload"):
          self.model = self.model.merge_and_unload()
          print("LoRA layers have been merged and unloaded.")
      else:
          raise AttributeError("The model does not support merge_and_unload.")


class MultitaskClassifier(ToxicityClassifier):
    """
    Multitask Classifier to handle multiple classification tasks.
    Extends the functionalities of ToxicityClassifier.
    """
    def __init__(self, config):
        super(MultitaskClassifier, self).__init__(config)

        # Add dropout layers
        self.translation_dropout = nn.Dropout(p=config.dropout_rate)

        # Norm Layer
        self.toxic_norm = nn.LayerNorm(self.model.config.hidden_size)

        self.translation_norm = nn.LayerNorm(self.model.config.hidden_size)

        # Add an additional classifier head for the second task
        self.translation_cls = nn.Linear(self.model.config.hidden_size, config.num_classes_translation_task)

        # Initialize new layers
        self.apply(initialize_weights)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output_pooled = output['pooler_output']
        return output_pooled

    def predict_toxicity(self, input_ids, attention_mask):
        """
        Predict the toxicity of the input text.
        """
        x = self.forward(input_ids, attention_mask)
        x = self.toxic_norm(x)
        x = self.toxic_dropout(x)
        logits = self.toxic_cls(x)
        return logits

    def predict_translation_id(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        """
        Predict if two sentences are translations of each other.
        """
        x = torch.cat((input_ids_1, input_ids_2), dim=1)
        att_m = torch.cat((attention_mask_1, attention_mask_2), dim=1)
        x = self.forward(x, att_m)
        x = self.translation_norm(x)
        x = self.translation_dropout(x)
        logits = self.translation_cls(x)
        return logits
    
    def extract_embeddings(self, input_ids, attention_mask):
        """
        Shared embedding extraction for both tasks.
        """
        outputs = self.model(input_ids, attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return cls_output