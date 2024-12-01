from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.toxic_dataset import (
    ToxicityDataset
)
from src.translation_dataset import (
    TranslationDataset
)
from src.utils import save_model, model_eval, save_losses

from src.classifier import MultitaskClassifier

def run():

    with open('train_config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
        config = SimpleNamespace(**config_dict)

    model_name = config.model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_small_sample = config.use_small_sample  # Add this option in your config
    small_sample_size = config.small_sample_size  # Add this to the config too
    val_sample_size = config.val_sample_size  # Add this to the config too

    #--------------- First Section: Model Initialization and Data Preparation ---------------#
    # Model
    model = MultitaskClassifier(config).to(device)

    # Training Data
    train_data_toxicity = ToxicityDataset(model.tokenizer, split='train', langs=config.language, local_file_path=None)
    train_data_translation = TranslationDataset(model.tokenizer, split='train', lang_pairs=config.lang_pairs, local_file_path=None)

    # Validation Data
    val_data_toxicity = ToxicityDataset(model.tokenizer, split='validation', langs=config.language, local_file_path=None)
    val_data_translation = TranslationDataset(model.tokenizer, split='dev', lang_pairs=config.lang_pairs, local_file_path=None)

    if use_small_sample:
        train_data_toxicity_subset = Subset(train_data_toxicity, range(min(small_sample_size, len(train_data_toxicity))))
        train_data_translation_subset = Subset(train_data_translation, range(min(small_sample_size, len(train_data_translation))))
        val_data_toxicity_subset = Subset(val_data_toxicity, range(min(val_sample_size, len(val_data_toxicity))))
        val_data_translation_subset = Subset(val_data_translation, range(min(val_sample_size, len(val_data_translation))))
    else:
        train_data_toxicity_subset = train_data_toxicity
        train_data_translation_subset = train_data_translation
        val_data_toxicity_subset = val_data_toxicity
        val_data_translation_subset = val_data_translation

    # DataLoaders
    train_loader_toxicity = DataLoader(
        train_data_toxicity_subset, batch_size=config.batch_size, shuffle=True, collate_fn=train_data_toxicity.collate_fn
    )
    val_loader_toxicity = DataLoader(
        val_data_toxicity_subset, batch_size=config.batch_size, shuffle=False, collate_fn=val_data_toxicity.collate_fn
    )
    train_loader_translation = DataLoader(
        train_data_translation_subset, batch_size=config.batch_size, shuffle=True, collate_fn=train_data_translation.collate_fn
    )
    val_loader_translation = DataLoader(
        val_data_translation_subset, batch_size=config.batch_size, shuffle=False, collate_fn=val_data_translation.collate_fn
    )
    #---------------------------------------------------------------------------------------#

    #------------------------ Second Section: Model Training --------------------------------#
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_file = open(f"loss_values_per_epoch_{model_name}.txt", "w")
    best_f1 = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch_toxicity, batch_translation in tqdm(
            zip(train_loader_toxicity, train_loader_translation), 
            total=min(len(train_loader_toxicity), len(train_loader_translation))
        ):
            
            optimizer.zero_grad()

            # Toxicity Task
            b_ids = batch_toxicity['token_ids']
            b_mask = batch_toxicity['attention_mask']
            b_labels = batch_toxicity['class_ids']
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            logits = model.predict_toxicity(b_ids, b_mask) 
            toxicity_loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float(), reduction='mean')

            # Translation Task
            b_ids = batch_translation['token_ids']    
            b_mask = batch_translation['attention_mask']
            b_ids_2 = batch_translation['non_en_token_ids']
            b_mask_2 = batch_translation['non_en_attention_mask']
            b_labels = batch_translation['class_ids']
            b_ids, b_mask, b_ids_2, b_mask_2, b_labels = b_ids.to(device), b_mask.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)
            logits = model.predict_translation_id(b_ids, b_mask, b_ids_2, b_mask_2)
            translation_loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float(), reduction='mean')
        
            save_losses(translation_loss, toxicity_loss, f"loss_log_file_{model_name}.txt", num_batches, epoch)
            
            combined_loss = toxicity_loss + translation_loss

            combined_loss.backward()

            optimizer.step()

            train_loss += combined_loss
            num_batches += 1

        train_loss = train_loss / (num_batches)
    #---------------------------------------------------------------------------------------#

    #------------------------ Third Section: Model Evaluation -------------------------------#
        # Evaluation on Training Data for each task
        print(f"Epoch: {epoch} | Loss: {train_loss}") 
        
        print("Translation Task")
        train_acc, train_precision, train_recall, train_f1 = model_eval(train_loader_translation, model, device, task='translation')
        print(f"Train | Acc: {train_acc} | F1: {train_f1} | Precision: {train_precision} | Recall: {train_recall}")
        
        # Evaluation on Validation Data
        val_acc, val_precision, val_recall, val_f1 = model_eval(val_loader_translation, model, device, task='translation')
        print(f"Validation | Acc: {val_acc} | F1: {val_f1} | Precision: {val_precision} | Recall: {val_recall}")

        print("Toxicity Task")
        train_acc, train_precision, train_recall, train_f1 = model_eval(train_loader_toxicity, model, device)
        print(f"Train | Acc: {train_acc} | F1: {train_f1} | Precision: {train_precision} | Recall: {train_recall}")
       
        # Evaluation on Validation Data
        val_acc, val_precision, val_recall, val_f1 = model_eval(val_loader_toxicity, model, device)
        print(f"Validation | Acc: {val_acc} | F1: {val_f1} | Precision: {val_precision} | Recall: {val_recall}")

        # Define model checkpointing
        if val_f1[0] > best_f1:
            print(f"Saving model with F1: {val_f1}")
            best_f1 = val_f1[0]
            save_model(model, config.model_output_path, model_name)

        # Write loss value to file
        loss_file.write(f"{epoch},{train_loss},{val_acc},{val_precision},{val_recall},{train_acc},{train_precision},{train_recall}\n")

if __name__ == '__main__':
    run()
