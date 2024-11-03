from types import SimpleNamespace

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_model, extract_embeddings
from src.toxic_dataset import PolygloToxicityBenchmark
from src.clustering import create_clusters
from src.plotting import plot_toxicity_distribution, plot_histogram

LANGUAGES = ['en', 'pt', 'es', 'de', 'nl', 'cs', 'pl']

def main():
    # Load Config
    with open('eval_config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
        config = SimpleNamespace(**config_dict)
    model_path = config.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Toxicity classifier
    model = load_model(model_path, config, device)

    # Load Benchmark Dataset for all laguages
    for lang in LANGUAGES:
        benchmark = PolygloToxicityBenchmark(model.tokenizer, lang=lang, split='ptp', sub_split='small', moderation='balanced')
        benchmark_loader = DataLoader(benchmark, batch_size=config.batch_size, shuffle=False, collate_fn=benchmark.collate_fn_for_clustering)

        # Extract embeddings for the Benchmark Dataset
        embeddings, texts, prompt_toxicity = extract_embeddings(model, benchmark_loader, device)

        # Create clusters using FAISS
        df_clusters = create_clusters(embeddings, texts, prompt_toxicity, num_clusters=3, device=device)
        print(df_clusters['prompt_toxicity'].describe())

        # Plot the distribution of prompt toxicity scores in each cluster
        filename = f'toxicity_distribution_{config.model_name}_{lang}.png'
        plot_toxicity_distribution(df_clusters, filename=filename)


if __name__ == '__main__':
    main()