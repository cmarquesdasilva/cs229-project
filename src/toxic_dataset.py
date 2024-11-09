from datasets import load_dataset, concatenate_datasets, Dataset
import torch

class ToxicityDataset(Dataset):
    def __init__(self, tokenizer, split, lang=None, local_file_path=None):
        self.tokenizer = tokenizer

        # Load dataset based on split and language
        self.dataset = self.fecth_balanced_toxi_text(split, lang)

        if split == "train":
            extra_datasets = []
            extra_datasets.append(load_dataset("textdetox/multilingual_toxicity_dataset", split=lang))
            if lang == "en":
                jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split=split)
                jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
                extra_datasets.append(jigsaw_dataset)
            self.dataset = concatenate_datasets([self.dataset] + extra_datasets)

        else:
            if lang == "en":
                jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split=split)
                jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
                self.dataset = concatenate_datasets([self.dataset,  jigsaw_dataset])

        # Load local dataset if provided
        if local_file_path:
            extra_datasets.append(load_dataset(data_files=local_file_path))

    def fecth_balanced_toxi_text(self, split, lang):
        # Load the dataset
        ds = load_dataset("FredZhang7/toxi-text-3M", split=split, save_infos=False, verification_mode='no_checks')

        # Filter by Language
        lang_filtered_ds = ds.filter(lambda x: x["lang"] == lang)
        
        # Separate Toxic and Non-Toxic Samples
        toxic_samples = lang_filtered_ds.filter(lambda x: x["is_toxic"] == 1)
        non_toxic_samples = lang_filtered_ds.filter(lambda x: x["is_toxic"] == 0)

        # Balance the Dataset by Sampling
        min_count = min(len(toxic_samples), len(non_toxic_samples)) # for test and val split

        # Randomly sample min_count examples from both toxic and non-toxic sets
        balanced_toxic_samples = toxic_samples.select(range(min_count))
        balanced_non_toxic_samples = non_toxic_samples.shuffle(seed=42).select(range(min_count))

        # Rename is_toxic to toxic
        balanced_toxic_samples = balanced_toxic_samples.rename_column("is_toxic", "toxic")
        balanced_non_toxic_samples = balanced_non_toxic_samples.rename_column("is_toxic", "toxic")

        # Concatenate and Shuffle the Balanced Dataset
        balanced_dataset = concatenate_datasets([balanced_toxic_samples, balanced_non_toxic_samples]).shuffle(seed=42)
        return balanced_dataset


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
        prompt_toxicity = [x['toxicity'] for x in data]

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