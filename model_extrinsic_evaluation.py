import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from types import SimpleNamespace
from src.utils import load_model, model_eval
from src.toxic_dataset import ToxicityDataset
from src.benchmark_dataset import PolygloToxicityBenchmark

BENCHMARK_LANGUAGES = ['en', 'es', 'pt', 'de', 'nl']
EVAL_LANGUAGES = ['en', 'es','de']

def main():
    results = {}
    # Load Config
    with open('eval_config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)
        config = SimpleNamespace(**config_dict)
    model_path = config.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_type = config.eval_type

    # Load Toxicity classifier
    if 'multitask' not in config.model_name:
        print('Loading single task model')
        model = load_model(model_path, config, device)
    else:
        model = load_model(model_path, config, device, model_type='multitask')

    # Load Benchmark Dataset for each language
    if eval_type == "validation":
        for lang in EVAL_LANGUAGES:
            val_data = ToxicityDataset(model.tokenizer, langs=lang, split='validation')
            val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=val_data.collate_fn)
            val_acc, val_precision, val_recall, val_f1 = model_eval(val_loader, model, device)
            results[lang] = {
                'val_accuracy': val_acc,
                'val_precision_safe': val_precision[0],
                'val_precision_toxic': val_precision[1],
                'val_recall_safe': val_recall[0],
                'val_recall_toxic': val_recall[1],
                'val_f1_safe': val_f1[0],
                'val_f1_toxic': val_f1[1],
            }
    else:
        for lang in BENCHMARK_LANGUAGES:
            benchmark = PolygloToxicityBenchmark(model.tokenizer, lang=lang, split='ptp', sub_split='small', moderation='balanced')
            benchmark_loader = DataLoader(benchmark, batch_size=32, shuffle=False, collate_fn=benchmark.collate_fn)
            benchmark_acc, benchmark_precision, benchmark_recall, benchmark_f1 = model_eval(benchmark_loader, model, device)
            results[lang] = {
                'benchmark_accuracy': benchmark_acc,
                'benchmark_precision_safe': benchmark_precision[0],
                'benchmark_precision_toxic': benchmark_precision[1],
                'benchmark_recall_safe': benchmark_recall[0],
                'benchmark_recall_toxic': benchmark_recall[1],
                'benchmark_f1_safe': benchmark_f1[0],
                'benchmark_f1_toxic': benchmark_f1[1]}

        # Create a DataFrame from the results dictionary
        df_results = pd.DataFrame.from_dict(results, orient='index')

        # Save the DataFrame to a CSV file
        df_results.to_csv(f'evaluation_results_{config.model_name}.csv', index=True, encoding='utf-8')

if __name__ == '__main__':
    main()