from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
import pandas as pd
import torch
from tqdm import tqdm


def load_local_dataset(file_path):
    """Load the local dataset from a CSV file."""
    df = pd.read_csv(file_path)
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading local dataset"):
        dataset.append({
            'text': row['comment_text'],
            'toxic': row['toxic']
        })
    return dataset


class ToxicityDataset(Dataset):
    def __init__(self, tokenizer, split, lang=None, local_file_path=None):
        self.tokenizer = tokenizer
        self.dataset = []

        # Load dataset based on split and lang
        if split == "train":
            if lang == "en":
                self.dataset.extend(load_dataset("textdetox/multilingual_toxicity_dataset", split=lang))
                jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split="validation")
                jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
                self.dataset.extend(jigsaw_dataset)
            else:
                self.dataset.extend(load_dataset("textdetox/multilingual_toxicity_dataset", split=lang))
        elif split == "validation" and lang == "en":
            jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split="validation")
            jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
            self.dataset.extend(jigsaw_dataset)
        elif split == "test" and lang == "en":
            jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split="test")
            jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
            self.dataset.extend(jigsaw_dataset)

        # Load local dataset if provided
        if local_file_path:
            self.dataset.extend(load_local_dataset(local_file_path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        texts = [x['text'] for x in data]
        class_ids = [x['toxic'] for x in data]

        encoding = self.tokenizer(texts, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, texts, class_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, texts, class_ids = self.pad_data(all_data)
        
        class_ids = torch.tensor(class_ids, dtype=torch.float)
        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'texts': texts,
                'class_ids': class_ids
            }

        return batched_data  
    

class PolygloToxicityBenchmark(Dataset):
    """
    Paper Reference: https://arxiv.org/pdf/2405.09373
    """
    def __init__(self, tokenizer, lang=None, split="wildchat", sub_split='wildchat', moderation="balanced"):
        self.tokenizer = tokenizer
        self.dataset = self.add_toxic_label(split, sub_split, moderation, lang)

    def add_toxic_label(self, split, sub_split, moderation, lang):
        """Add 'toxic' key based on 'prompt_toxicity'."""

        # load dataset
        data = load_dataset("ToxicityPrompts/PolygloToxicityPrompts", f"{split}-{lang}")

        if moderation == "strict":
            threshold = 0.5
        elif moderation == "balanced":
            threshold = 0.7
        elif moderation == "relaxed":
            threshold = 0.85
        else:
            raise ValueError("Invalid moderation level. Choose from 'strict', 'balanced', or 'relaxed'.")
        
        # Define a function for mapping 'toxic' labels
        def label_toxicity(example):
            if sub_split == "wildchat":
                example['toxic'] = 1 if example['prompt_toxicity'] > threshold else 0
            else:
                example['toxic'] = 1 if example['toxicity'] > threshold else 0
            return example

        # Apply the function to the dataset
        if sub_split == "wildchat":
            data[split] = data[split].map(label_toxicity)

            return data[split]
        else:
            data[sub_split] = data[sub_split].map(label_toxicity)

            return data[sub_split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        texts = [x['prompt'] for x in data]
        class_ids = [x['toxic'] for x in data]

        encoding = self.tokenizer(texts, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, texts, class_ids

    def pad_data_for_clustering(self, data):
        texts = [x['prompt'] for x in data]
        prompt_toxicity = [x['prompt_toxicity'] for x in data]

        encoding = self.tokenizer(texts, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, prompt_toxicity, texts
    
    def collate_fn(self, all_data):
        token_ids, attention_mask, texts, class_ids = self.pad_data(all_data)
        
        class_ids = torch.tensor(class_ids, dtype=torch.float)

        batched_data = {
               'token_ids': token_ids,
               'attention_mask': attention_mask,
               'texts': texts,
               'class_ids': class_ids
        }
        return batched_data

    def collate_fn_for_clustering(self, all_data):
        token_ids, attention_mask, prompt_toxicity, texts = self.pad_data_for_clustering(all_data)
        
        prompt_toxicity = torch.tensor(prompt_toxicity, dtype=torch.float)

        batched_data = {
               'token_ids': token_ids,
               'attention_mask': attention_mask,
               'texts': texts,
               'prompt_toxicity': prompt_toxicity
        }
        return batched_data