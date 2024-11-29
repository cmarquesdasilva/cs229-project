from datasets import load_dataset, concatenate_datasets, Dataset
import torch
import random

class TranslationDataset(Dataset):

    DATASET_NAME = "sentence-transformers/parallel-sentences-talks"
    
    def __init__(self, tokenizer, split, lang_pairs=None, local_file_path=None):
        """
        Initializes the dataset for translation identification with multilingual support.
        
        Args:
            tokenizer: The tokenizer used to encode text data.
            split (str): The dataset split to load ('train', 'test', or 'validation').
            langs (list or str): Languages for the dataset, can be a single language (str) or a list of languages.
            local_file_path (str): Optional path to a local dataset file.
        """
        self.tokenizer = tokenizer
        self._split = split
        if isinstance(lang_pairs, str):
            self.lang_pairs = lang_pairs.split(",") if "," in lang_pairs else [lang_pairs]
        else:
            self.lang_pairs = lang_pairs
        self.dataset = self.prepare_multilingual_dataset(local_file_path)

    def prepare_multilingual_dataset(self, local_file_path):
        """Prepares and returns a multilingual dataset by merging datasets for each specified language."""
        language_datasets_positives = [self.create_label_column(lang_pair) for lang_pair in self.lang_pairs]
        language_datasets_negatives = [self.create_label_column(lang_pair, label_class='negative') for lang_pair in self.lang_pairs]
        combined_dataset = concatenate_datasets(language_datasets_positives + language_datasets_negatives)
        return combined_dataset.shuffle(seed=42)

    def create_label_column(self, lang_pair, label_class='positive'):
        # Load the dataset
        ds = load_dataset(TranslationDataset.DATASET_NAME, lang_pair)
        ds = ds[self._split]
        # Shuffle the column "english"     
        if label_class == 'negative':
            # Extract and shuffle the "english" column
            english_column = ds['english']
            shuffled_english = english_column.copy()
            random.seed(42)
            random.shuffle(shuffled_english)

            # Remove the original "english" column
            ds = ds.remove_columns('english')

            # Add the shuffled "english" column
            ds = ds.add_column('english', shuffled_english)

            # Create a new column "label" with value 0
            ds = ds.map(lambda example: {**example, "label": 0})
            return ds
        else:
            # Create a new column "label" with value 1
            ds = ds.map(lambda example: {**example, "label": 1})
            return ds
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        en_sentence = [x['english'] for x in data]
        non_en_sentence = [x['non_english'] for x in data]
        class_ids = [x['label'] for x in data]

        en_encoding = self.tokenizer(en_sentence, return_tensors='pt', padding="max_length", truncation=True, max_length=256)
        token_ids = torch.LongTensor(en_encoding['input_ids'])
        attention_mask = torch.LongTensor(en_encoding['attention_mask'])

        non_en_enconding = self.tokenizer(non_en_sentence, return_tensors='pt', padding="max_length", truncation=True, max_length=256)
        non_en_token_ids = torch.LongTensor(non_en_enconding['input_ids'])
        non_en_attention_mask = torch.LongTensor(non_en_enconding['attention_mask'])

        return token_ids, attention_mask, non_en_token_ids, non_en_attention_mask, class_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, non_en_token_ids, non_en_attention_mask, class_ids = self.pad_data(all_data)
        class_ids = torch.tensor(class_ids, dtype=torch.float)
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'non_en_token_ids': non_en_token_ids,
            'non_en_attention_mask': non_en_attention_mask,
            'class_ids': class_ids
        }