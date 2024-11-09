import os
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm import tqdm
from .roberta_clf import ToxicityClassifier
from peft import PeftModel, PeftConfig

# Helper Functions
def check_label_distribution(dataloader, device):
    if device.type == 'cpu':
        return check_label_distribution_cpu(dataloader)
    else:
        return check_label_distribution_gpu(dataloader)

def check_label_distribution_cpu(dataloader):
    label_counts = Counter()
    for batch in dataloader:
        labels = batch['class_ids'].cpu().numpy()
        label_counts.update(labels)
    return label_counts

def check_label_distribution_gpu(dataloader):
    unique_labels, counts = None, None
    for batch in dataloader:
        labels = batch['class_ids'] 
        if unique_labels is None:
            unique_labels, counts = torch.unique(labels, return_counts=True)
        else:
            batch_labels, batch_counts = torch.unique(labels, return_counts=True)
            counts = counts + torch.scatter_add(torch.zeros_like(counts), 0, batch_labels, batch_counts)
    return dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))

def save_model(model, path, model_name):
    """Save the model to the specified path."""
    full_path = os.path.join(path, f"{model_name}.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {path}")


def load_model(model_path, config, device):
    """Load the model from the specified path."""
    model = ToxicityClassifier(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    if config.option == 'lora':
      model.merge_lora()
    
    model.to(device)
    model.eval()
    return model

def model_eval(dataloader, model, device, output_file=None):

    model.eval() # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    texts = []

    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=False)):
        b_ids, b_mask, b_texts, b_labels = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['texts'], batch['class_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = (logits > 0).astype(int).flatten() #np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        texts.extend(b_texts)

    f1 = f1_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)

    if output_file:
        with open(output_file, 'w') as f:
            for i in range(len(y_true)):
                error = 0 if y_true[i] == y_pred[i] else 1
                f.write(f"{texts[i]}\t{y_true[i]}\t{y_pred[i]}\t{error}\n")
    
    return acc, precision, recall, f1


def extract_embeddings(model, dataloader, device):
    embeddings = []
    texts = []
    prompt_toxicity = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_texts = batch['texts']
            batch_prompt_toxicity = batch['prompt_toxicity']

            batch_embeddings = model.extract_embeddings(input_ids, attention_mask)
            embeddings.append(batch_embeddings)
            texts.extend(batch_texts)
            prompt_toxicity.extend(batch_prompt_toxicity)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, texts, prompt_toxicity