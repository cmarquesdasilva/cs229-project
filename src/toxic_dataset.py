"""
This script defines the ToxicityDataset class, which is used to load and process a multilingual dataset for toxicity classification.
The dataset can be loaded from multiple languages and supports caching of dataset splits to improve performance.
"""
from datasets import load_dataset, concatenate_datasets, Dataset
import torch


class ToxicityDataset(Dataset):
    _cached_splits = {}  # Class-level cache for splits

    def __init__(self, tokenizer, split, langs=None, seed=42, validation_split_ratio=0.2):
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
        self.seed = seed
        self.validation_split_ratio = validation_split_ratio
        if isinstance(langs, str):
            self.langs = langs.split(",") if "," in langs else [langs]
        else:
            self.langs = langs
        self.dataset = self.prepare_multilingual_dataset()

    def prepare_multilingual_dataset(self):
        """Prepares and returns a multilingual dataset by merging datasets for each specified language and
        remove contamination.
        """
        # Generate a unique key for this split configuration
        cache_key = (tuple(self.langs), self.validation_split_ratio, self.seed)

        if cache_key not in ToxicityDataset._cached_splits:
            print('Creating cache split...')
            multilingual_set = self.add_multilingual_datasets()

            train_dataset, validation_dataset = multilingual_set.train_test_split(
                test_size=self.validation_split_ratio, seed=self.seed
            ).values()

            training_toxic_set = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="balanced_train")
            training_toxic_set = training_toxic_set.map(self.convert_labels)
            training_toxic_set = training_toxic_set.remove_columns("label")
            training_toxic_set = training_toxic_set.rename_column("comment_text", "text")
            combined_training_dataset = concatenate_datasets([training_toxic_set, train_dataset])

            validation_toxic_set = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="validation")
            validation_toxic_set = validation_toxic_set.map(self.convert_labels)
            validation_toxic_set = validation_toxic_set.remove_columns("label")
            validation_toxic_set = validation_toxic_set.rename_column("comment_text", "text")
            combined_validation_dataset = concatenate_datasets([validation_toxic_set, validation_dataset])

            combined_validation_dataset = self.remove_validation_contamination(combined_training_dataset, combined_validation_dataset)

            ToxicityDataset._cached_splits[cache_key] = (combined_training_dataset, combined_validation_dataset)
        else:
            print('Using cached split...')
            train_dataset, validation_dataset = ToxicityDataset._cached_splits[cache_key]

        # Select the appropriate split
        if self._split == "train":
            combined_dataset = train_dataset
        elif self._split == "validation":
            combined_dataset = validation_dataset
        else:
            raise ValueError("Invalid split. Choose from 'train' or 'validation'.")

        return combined_dataset.shuffle(seed=42)

    # Define a function to convert labels
    def convert_labels(self, ds):
        ds['toxic'] = 1 if ds['label'] == 'tox' else 0
        return ds
    
    def add_multilingual_datasets(self):
        """Adds additional datasets for training or specific languages."""
        extra_datasets = []
        for lang in self.langs:
            if lang in ['en', 'ru', 'uk', 'de', 'es', 'am', 'zh', 'ar', 'hi']:
                extra_datasets.append(load_dataset("textdetox/multilingual_toxicity_dataset", split=lang))
            if lang == "en":
                jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split=self._split)
                jigsaw_dataset = jigsaw_dataset.rename_column("comment_text", "text")
                extra_datasets.append(jigsaw_dataset)
        return concatenate_datasets(extra_datasets)
    
    def remove_validation_contamination(self, train_dataset, validation_dataset):
        """
        Removes overlapping examples from the validation dataset to ensure no contamination.

        Args:
            train_dataset: The training dataset (a Hugging Face Dataset object).
            validation_dataset: The validation dataset (a Hugging Face Dataset object).

        Returns:
            validation_dataset: A filtered validation dataset with no contamination.
        """
        # Extract text examples from the training dataset
        train_texts = set(example['text'] for example in train_dataset)

        # Filter the validation dataset
        filtered_validation = validation_dataset.filter(lambda example: example['text'] not in train_texts)
        print(len(filtered_validation), len(validation_dataset))
        return filtered_validation
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def to_pandas(self):
        return self.dataset.to_pandas()
    
    def add_predictions(self, predictions):
        """
        Adds predictions to the dataset as a new column.

        Args:
            predictions (list): A list of predictions to add to the dataset.
        """
        # Ensure predictions length matches the dataset
        if len(predictions) != len(self.dataset):
            raise ValueError("Length of predictions must match the length of the dataset.")

        # Add the predictions column to the dataset
        self.dataset = self.dataset.add_column("Prediction", predictions)


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
    
    def add_new_examples(self, new_examples):
        """
        Adds new examples to the existing dataset.

        Args:
            new_examples: A dictionary containing new examples.
        """
        new_examples_dataset = Dataset.from_dict(new_examples)
        self.dataset = concatenate_datasets([self.dataset, new_examples_dataset])