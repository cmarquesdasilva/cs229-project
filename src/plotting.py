import matplotlib.pyplot as plt
import seaborn as sns

def plot_toxicity_distribution(df, filename='toxicity_distribution.png'):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='prompt_toxicity', data=df)
    plt.axhline(y=0.5, color='blue', linestyle='--', label='y = 0.5')
    plt.axhline(y=0.7, color='purple', linestyle='--', label='y = 0.7')
    plt.axhline(y=0.85, color='red', linestyle='--', label='y = 0.85')
    plt.title('Distribution of Prompt Toxicity Scores in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Prompt Toxicity')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['prompt_toxicity'], bins=20, kde=True)
    plt.title('Distribution of Prompt Toxicity Scores')
    plt.xlabel('Prompt Toxicity')
    plt.ylabel('Frequency')
    plt.show()