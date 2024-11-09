import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from types import SimpleNamespace
from src.utils import load_model, model_eval
from src.toxic_dataset import PolygloToxicityBenchmark, ToxicityDataset

LANGUAGES = ['en', 'pt', 'es', 'de', 'nl']

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
        val_data = ToxicityDataset(model.tokenizer, lang=lang, split='validation')
        test_data = ToxicityDataset(model.tokenizer, lang=lang, split='test')
        benchmark_loader = DataLoader(benchmark, batch_size=32, shuffle=False, collate_fn=benchmark.collate_fn)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=val_data.collate_fn)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=test_data.collate_fn)
        
        # model evaluation process
        benchmark_acc, benchmark_precision, benchmark_recall, benchmark_f1 = model_eval(benchmark_loader, model, device)
        val_acc, val_precision, val_recall, val_f1 = model_eval(val_loader, model, device)
        test_acc, test_precision, test_recall, test_f1 = model_eval(test_loader, model, device)


        results[lang] = {
            'benchmark_accuracy': benchmark_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'benchmark_precision_safe': benchmark_precision[0],
            'benchmark_precision_toxic': benchmark_precision[1],
            'val_precision_safe': val_precision[0],
            'val_precision_toxic': val_precision[1],
            'test_precision_safe': test_precision[0],
            'test_precision_toxic': test_precision[1],
            'benchmark_recall_safe': benchmark_recall[0],
            'benchmark_recall_toxic': benchmark_recall[1],
            'val_recall_safe': val_recall[0],
            'val_recall_toxic': val_recall[1],
            'test_recall_safe': test_recall[0],
            'test_recall_toxic': test_recall[1],
            'benchmark_f1_safe': benchmark_f1[0],
            'benchmark_f1_toxic': benchmark_f1[1],
        }

        print(f"Language: {lang}")
        print(f"Accuracy: {benchmark_acc}, {val_acc}, {test_acc}")
        print(f"Precision: {benchmark_precision}, {val_precision}, {test_precision}")
        print(f"Recall: {benchmark_recall}, {val_recall}, {test_recall}")

        # Create a DataFrame from the results dictionary
        df_results = pd.DataFrame.from_dict(results, orient='index')

        # Save the DataFrame to a CSV file
        df_results.to_csv(f'evaluation_results_{config.model_name}.csv', index=True, encoding='utf-8')

if __name__ == '__main__':
    main()