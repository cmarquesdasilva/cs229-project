import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from types import SimpleNamespace
from src.utils import load_model, model_eval
from src.toxic_dataset import PolygloToxicityBenchmark

LANGUAGES = ['en', 'pt', 'es', 'de', 'nl', 'cs', 'pl']

def main():
    results = {}
    # Load Config
    with open('eval_config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
        config = SimpleNamespace(**config_dict)
    model_path = config.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Toxicity classifier
    model = load_model(model_path, config, device)

    # Load Benchmark Dataset for each language
    for lang in LANGUAGES:
        benchmark = PolygloToxicityBenchmark(model.tokenizer, lang=lang, split='ptp', sub_split='small', moderation='balanced')
        benchmark_loader = DataLoader(benchmark, batch_size=32, shuffle=False, collate_fn=benchmark.collate_fn)

        # model evaluation process
        benchmark_acc, benchmark_precision, benchmark_recall, benchmark_f1 = model_eval(benchmark_loader, model, device)
        
        results[lang] = {
            'accuracy': benchmark_acc,
            'precision': benchmark_precision,
            'recall': benchmark_recall,
            'f1': benchmark_f1
        }

        print(f"Language: {lang}")
        print(f"Accuracy: {benchmark_acc}")
        print(f"Precision: {benchmark_precision}")
        print(f"Recall: {benchmark_recall}")

        # Create a DataFrame from the results dictionary
        df_results = pd.DataFrame.from_dict(results, orient='index')

        # Save the DataFrame to a CSV file
        df_results.to_csv('benchmark_results.csv', index=True, encoding='utf-8')

if __name__ == '__main__':
    main()