from types import SimpleNamespace

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_model, extract_embeddings
from src.toxic_dataset import ToxicityDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

LANGUAGES = ['en'] #, 'es']

def main():
    # Load Config
    with open('eval_config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
        config = SimpleNamespace(**config_dict)
    model_path = config.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model_type = config.model_name.split('_')[2]
    
    # Load Toxicity classifier
    model = load_model(model_path, config, device, model_type="multitask")

    # Load Benchmark Dataset for all laguages
    for lang in LANGUAGES:
        val_data = val_data = ToxicityDataset(model.tokenizer, langs=lang, split='validation')
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=val_data.collate_fn)

        # Extract embeddings for the Benchmark Dataset
        embeddings, texts, prompt_toxicity = extract_embeddings(model, val_loader, 'class_ids',  device)

        # Reduce Embedding Dimension using PCA for visualization
        embeddings = embeddings.cpu().numpy()
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot the data points and colorized then using the column "prompt_toxicity"
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=prompt_toxicity, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='Prompt Toxicity')
        plt.title('PCA of Embeddings')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        # Save the plot
        filename = f'pca_embeddings_{config.model_name}_{lang}.png'
        plt.savefig(filename)

if __name__ == '__main__':
    main()