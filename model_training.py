from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
)
from src.toxic_dataset import (
    ToxicityDataset,
)
from src.utils import check_label_distribution, save_model, model_eval

from src.roberta_clf import ToxicityClassifier, train_model

def run():

    with open('train_config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
        config = SimpleNamespace(**config_dict)

    batch_size = config.batch_size
    langs = config.language
    model_name = config.model_name
    lr = config.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = ToxicityClassifier(config).to(device)

    # Load Training Data
    train_data = ToxicityDataset(model.tokenizer, split='train', lang=langs, local_file_path=None)
    val_data = ToxicityDataset(model.tokenizer, split='validation', lang=langs, local_file_path=None)

    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=val_data.collate_fn)

    # Check label distribution
    #train_label_distribution = check_label_distribution(train_loader, device)
    #val_label_distribution = check_label_distribution(val_loader,device)

    #print(f"Train Label Distribution: {train_label_distribution}")
    #print(f"Validation Label Distribution: {val_label_distribution}")

    # Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Reduce LR by a factor of 0.9 every epoch
    loss_file = open(f"loss_values_{model_name}.txt", "w")
    best_f1 = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_loader):
            loss = train_model(model, batch, optimizer, device)
            train_loss += loss
            num_batches += 1

        # Step the scheduler at each iteration
        scheduler.step()

        train_loss = train_loss / (num_batches)

        train_acc, train_precision, train_recall, train_f1 = model_eval(train_loader, model, device)
        print(f"Epoch: {epoch} | Loss: {train_loss}") 
        print(f"Train | Acc: {train_acc} | F1: {train_f1} | Precision: {train_precision} | Recall: {train_recall}")

        # Evaluation on Validation Data
        val_acc, val_precision, val_recall, val_f1 = model_eval(val_loader, model, device)
        print(f"Validation | Acc: {val_acc} | F1: {val_f1} | Precision: {val_precision} | Recall: {val_recall}")

        # Define model checkpointing
        if val_f1[0] > best_f1:
            print(f"Saving model with F1: {val_f1}")
            best_f1 = val_f1[0]
            save_model(model, config.model_output_path, model_name)

        # Write loss value to file
        loss_file.write(f"{epoch},{train_loss},{val_acc},{val_precision},{val_recall},{train_acc},{train_precision},{train_recall}\n")

    # Test set
    #test_data = ToxicityDataset(model.tokenizer, lang="en", split = 'test', local_file_path=None)
    #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_data.collate_fn)
    #test_acc, test_precision, test_recall, test_f1 = model_eval(test_loader, model, device)
    #print(f"Test | Acc: {test_acc} | F1: {test_f1} | Precision: {test_precision} | Recall: {test_recall}")

if __name__ == '__main__':
    run()