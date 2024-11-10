from datasets import load_dataset, concatenate_datasets, Dataset
import torch

class ToxicityDataset(Dataset):
    def __init__(self, tokenizer, split, langs=None, local_file_path=None):
        """
        Initializes the dataset for toxicity classification with multilingual support.
        
        Args:
            tokenizer: The tokenizer used to encode text data.
            split (str): The dataset split to load ('train', 'test', or 'validation').
            langs (list or str): Languages for the dataset, can be a single language (str) or a list of languages.
            local_file_path (str): Optional path to a local dataset file.
        """
        self.tokenizer = tokenizer
        self._split = split
        self.langs = [langs] if isinstance(langs, str) else langs
        self.dataset = self.prepare_multilingual_dataset(local_file_path)

    def prepare_multilingual_dataset(self, local_file_path):
        """Prepares and returns a multilingual dataset by merging datasets for each specified language."""
        language_datasets = [self.fetch_balanced_toxi_text(lang) for lang in self.langs]
        combined_dataset = concatenate_datasets(language_datasets)

        # Add extra datasets based on split and language
        if self._split == "train":
            combined_dataset = self.add_extra_datasets(combined_dataset)
        
        # Include local dataset if provided
        if local_file_path:
            local_dataset = load_dataset("text", data_files=local_file_path)["train"]
            combined_dataset = concatenate_datasets([combined_dataset, local_dataset])

        return combined_dataset.shuffle(seed=42)

    def add_extra_datasets(self, base_dataset):
        """Adds additional datasets for training or specific languages."""
        extra_datasets = []
        for lang in self.langs:
            if lang in ['en', 'ru', 'uk', 'de', 'es', 'am', 'zh', 'ar', 'hi']:
                extra_datasets.append(load_dataset("textdetox/multilingual_toxicity_dataset", split=lang))
            if lang == "en":
                jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split=self._split)
                jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
                extra_datasets.append(jigsaw_dataset)
        return concatenate_datasets([base_dataset] + extra_datasets)

    def fetch_balanced_toxi_text(self, lang):
        """Fetches a balanced dataset with toxic and non-toxic samples for a specified language."""
        ds = load_dataset("FredZhang7/toxi-text-3M", split=self._split)
        ds_lang = ds.filter(lambda x: x["lang"] == lang)
        toxic_samples = ds_lang.filter(lambda x: x["is_toxic"] == 1)
        non_toxic_samples = ds_lang.filter(lambda x: x["is_toxic"] == 0)
        
        min_count = min(len(toxic_samples), len(non_toxic_samples))
        toxic_balanced = toxic_samples.select(range(min_count))
        non_toxic_balanced = non_toxic_samples.shuffle(seed=42).select(range(min_count))

        balanced_dataset = concatenate_datasets([toxic_balanced, non_toxic_balanced]).shuffle(seed=42)
        return balanced_dataset.rename_column("is_toxic", "toxic")

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
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'texts': texts,
            'class_ids': class_ids
        }

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