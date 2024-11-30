"""
This script defines the PolygloToxicityBenchmark class, which is used to load and process a dataset for toxicity benchmarking.
The dataset is loaded from the "ToxicityPrompts/PolygloToxicityPrompts" source, and a 'toxic' label is added based on the 
'prompt_toxicity' or 'toxicity' values, depending on the specified moderation level and sub-split.
"""
from datasets import load_dataset, Dataset
import torch

class PolygloToxicityBenchmark(Dataset):
    """
    Paper Reference: https://arxiv.org/pdf/2405.09373
    """
    def __init__(self, tokenizer, lang=None, split="wildchat", sub_split='wildchat', moderation="balanced"):
        """
        Args:
            tokenizer: The tokenizer to be used for processing the dataset.
            lang: The language of the dataset.
            split: The split of the dataset to be used.
            sub_split: The sub-split of the dataset to be used.
            moderation: The moderation level for labeling toxicity.
        """
        self.tokenizer = tokenizer
        self.dataset = self.add_toxic_label(split, sub_split, moderation, lang)

    def add_toxic_label(self, split, sub_split, moderation, lang):
        """
        Add 'toxic' key based on 'prompt_toxicity' or 'toxicity'.

        Args:
            split: The split of the dataset to be used.
            sub_split: The sub-split of the dataset to be used.
            moderation: The moderation level for labeling toxicity.
            lang: The language of the dataset.

        Returns:
            The dataset with the added 'toxic' labels.
        """
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
            """
            Label the examples in the dataset as toxic or non-toxic.

            Args:
                example: A single example from the dataset.

            Returns:
                The example with the added 'toxic' label.
            """
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
        """
        Pad the data to the maximum length and tokenize the texts.

        Args:
            data: The dataset to be padded and tokenized.

        Returns:
            token_ids: The tokenized input IDs.
            attention_mask: The attention mask for the tokenized inputs.
            texts: The original texts.
            class_ids: The class IDs for the toxicity labels.
        """
        texts = [x['prompt'] for x in data]
        class_ids = [x['toxic'] for x in data]

        encoding = self.tokenizer(texts, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, texts, class_ids
    
    def collate_fn(self, all_data):
        """
        Collate function to create batches of data.

        Args:
            all_data: The dataset to be collated into batches.

        Returns:
            batched_data: A dictionary containing batched token IDs, attention masks, texts, and class IDs.
        """
        token_ids, attention_mask, texts, class_ids = self.pad_data(all_data)
        
        class_ids = torch.tensor(class_ids, dtype=torch.float)

        batched_data = {
               'token_ids': token_ids,
               'attention_mask': attention_mask,
               'texts': texts,
               'class_ids': class_ids
        }
        return batched_data
