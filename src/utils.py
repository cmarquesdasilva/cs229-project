""""Utility functions for training and evaluation."""
import os
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from .classifier import ToxicityClassifier, MultitaskClassifier
import pandas as pd
from datasets import Dataset
from typing import Dict, Union

# Helper Functions
def check_label_distribution(dataloader: DataLoader, device: torch.device) -> Dict[int, int]:
    """
    Check the distribution of labels in the dataloader.

    Args:
        dataloader: The dataloader containing the dataset.
        device: The device (CPU or GPU) to be used for computation.

    Returns:
        A dictionary containing the count of each label.
    """
    if device.type == 'cpu':
        return check_label_distribution_cpu(dataloader)
    else:
        return check_label_distribution_gpu(dataloader)

def check_label_distribution_cpu(dataloader: DataLoader) -> Dict[int, int]:
    """
    Check the distribution of labels in the dataloader using CPU.

    Args:
        dataloader: The dataloader containing the dataset.

    Returns:
        A Counter object containing the count of each label.
    """ 
    label_counts = Counter()
    for batch in dataloader:
        labels = batch['class_ids'].cpu().numpy()
        label_counts.update(labels)
    return label_counts

def check_label_distribution_gpu(dataloader: DataLoader) -> Dict[int, int]:
    """
    Check the distribution of labels in the dataloader using GPU.

    Args:
        dataloader: The dataloader containing the dataset.

    Returns:
        A dictionary containing the count of each label.
    """
    unique_labels, counts = None, None
    for batch in dataloader:
        labels = batch['class_ids'] 
        if unique_labels is None:
            unique_labels, counts = torch.unique(labels, return_counts=True)
        else:
            batch_labels, batch_counts = torch.unique(labels, return_counts=True)
            counts = counts + torch.scatter_add(torch.zeros_like(counts), 0, batch_labels, batch_counts)
    return dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))

def save_model(model: nn.Module, path: str, model_name: str) -> None:
    """Save the model to the specified path."""
    full_path = os.path.join(path, f"{model_name}.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {path}")

def load_model(model_path: str, config: Dict[str,str], device: torch.device, model_type='Toxicity') -> nn.Module:
    """
    Load and initialize the model based on the model_type parameter.

    Args:
        model_type (str): The type of model to load ('Multitask' or other types).
        config (object): The configuration object for the model.

    Returns:
        model (nn.Module): The initialized model.
    """
    model_type = model_type.lower()
    if model_type == 'multitask':
        model = MultitaskClassifier(config)
        model.load_state_dict(torch.load(model_path, map_location=device))

    elif model_type == 'toxicity':
        model = ToxicityClassifier(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if config.option == 'lora':
      model.merge_lora()
    
    model.to(device)
    model.eval()
    return model

def model_eval(dataloader: DataLoader, model: nn.Module, device: torch.device, task: str = 'toxicity') -> tuple:
    """
    Evaluate the model on the specified task.

    Args:
        dataloader: The dataloader containing the dataset.
        model: The model to be evaluated.
        device: The device (CPU or GPU) to be used for computation.
        task: The task to evaluate the model on. Options are 'toxicity' or 'translation'.

    Returns:
        A tuple containing accuracy, precision, recall, and F1 score.
    """
    if task == 'toxicity':
        acc, precision, recall, f1 = model_eval_toxicity_task(dataloader, model, device)
    elif task == 'translation':
        acc, precision, recall, f1 = model_eval_translation_task(dataloader, model, device)
    else:
        raise ValueError(f"Task {task} not supported.")
    
    return acc, precision, recall, f1

def model_eval_translation_task(dataloader: DataLoader, model: nn.Module, device: torch.device) -> tuple:
    """
    Evaluate the model on the translation task.

    Args:
        dataloader: The dataloader containing the dataset.
        model: The model to be evaluated.
        device: The device (CPU or GPU) to be used for computation.

    Returns:
        A tuple containing accuracy, precision, recall, and F1 score.
    """ 
    model.eval()
    y_preds = [] 
    y_true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", disable=False):
            # Extract inputs
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_ids_2 = batch['non_en_token_ids'].to(device)
            b_mask_2 = batch['non_en_attention_mask'].to(device)  
            b_labels = batch['class_ids'].to(device).flatten()

            # Forward pass
            logits = model.predict_translation_id(b_ids, b_mask, b_ids_2, b_mask_2)
            preds = (logits > 0).long().view(-1).cpu().tolist()  # Thresholding logits
            b_labels = b_labels.cpu().tolist()  # Move labels to CPU

            # Accumulate results
            y_preds.extend(preds)
            y_true_labels.extend(b_labels)

    # Metrics calculation
    acc = accuracy_score(y_true_labels, y_preds)
    precision = precision_score(y_true_labels, y_preds, average=None)
    recall = recall_score(y_true_labels, y_preds, average=None)
    f1 = f1_score(y_true_labels, y_preds, average=None)

    return acc, precision, recall, f1

def model_eval_toxicity_task(dataloader, model, device):
    """
    Evaluate the model on the toxicity task.

    Args:
        dataloader: The dataloader containing the dataset.
        model: The model to be evaluated.
        device: The device (CPU or GPU) to be used for computation.

    Returns:
        A tuple containing accuracy, precision, recall, and F1 score.
    """
    model.eval()
    y_preds = []
    y_true_labels = [] 
    texts = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", disable=False):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_texts = batch['texts']
            b_labels = batch['class_ids'].to(device).flatten()

            # Forward pass
            if hasattr(model, 'predict_toxicity'):
                logits = model.predict_toxicity(b_ids, b_mask)
            else:
                logits = model(b_ids, b_mask)

            preds = (logits > 0).long().view(-1).cpu().tolist()
            b_labels = b_labels.cpu().tolist()

            # Accumulate results
            y_true_labels.extend(b_labels)
            y_preds.extend(preds)
            texts.extend(b_texts)

    # Metrics calculation
    acc = accuracy_score(y_true_labels, y_preds)
    precision = precision_score(y_true_labels, y_preds, average=None)
    recall = recall_score(y_true_labels, y_preds, average=None)
    f1 = f1_score(y_true_labels, y_preds, average=None)

    return acc, precision, recall, f1

def extract_embeddings(model: nn.Module, dataloader: DataLoader, label_column: str, device: torch.device) -> Dict[str, Union[torch.Tensor, list]]:
    """
    Extract embeddings from the model for the given dataloader.

    Args:
        model: The model used to extract embeddings.
        dataloader: The dataloader containing the dataset.
        label_column: The column name for the labels in the dataset.
        device: The device (CPU or GPU) to be used for computation.

    Returns:
        A dictionary containing the embeddings, texts, and prompt toxicity.
    """    
    embeddings = []
    texts = []
    prompt_toxicity = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_texts = batch['texts']
            batch_prompt_toxicity = batch[label_column]

            batch_embeddings = model.extract_embeddings(input_ids, attention_mask)
            embeddings.append(batch_embeddings)
            texts.extend(batch_texts)
            prompt_toxicity.extend(batch_prompt_toxicity)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, texts, prompt_toxicity

def save_losses(translation_loss, toxicity_loss, loss_log_file, batch_num, epoch):
    """
    Save the translation loss and toxicity loss after each batch update.

    Args:
        translation_loss (float): The loss value for the translation task.
        toxicity_loss (float): The loss value for the toxicity task.
        loss_log_file (str): The file path to save the loss values.
        batch_num (int): The current batch number.
        epoch (int): The current epoch number.
    """
    with open(loss_log_file, 'a') as f:
        f.write(f"Epoch: {epoch}, Batch: {batch_num}, Translation Loss: {translation_loss:.4f}, Toxicity Loss: {toxicity_loss:.4f}\n")

def store_predictions(dataloader: DataLoader, model: nn.Module, device: torch.device) -> pd.DataFrame:
    """
    Run model inference, store the predictions in the dataset

    Args:
        dataloader (DataLoader): The DataLoader for the dataset.
        model (nn.Module): The model to use for inference.
        device (torch.device): The device to run the model on.
    Returns:
        df (pd.DataFrame): The dataset with the predictions added.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", disable=False):
            # Extract inputs
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)

            # Forward pass
            if hasattr(model, 'predict_toxicity'):
                logits = model.predict_toxicity(b_ids, b_mask)
            else:
                logits = model(b_ids, b_mask)

            preds = (logits > 0).long().view(-1).cpu().tolist()
            predictions.extend(preds)

    dataset = dataloader.dataset
    dataset.add_predictions(predictions)
    df = dataset.to_pandas()
    return df