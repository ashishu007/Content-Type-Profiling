"""
Plot the evaluation results.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datasets = ['MLB', 'SportSett', 'SumTime', 'Obituary']

def save_df_dist_plot(df):
    """
    """
    plt.rcParams.update({'font.size': 12})
    ax = df.transpose().plot.bar(figsize=(9, 5), rot=0)#, marker='*', markersize=10, linewidth=3)
    for p in ax.patches:
        ax.annotate(f'{str(int(p.get_height()))}', (p.get_x() * 1.005, (p.get_height() * 1.005) + 2))
    ax.set_title(f'Evaluation Results')
    ax.set_ylim(0, 110)
    # ax.set_ylabel('Percentage')
    # ax.set_xticklabels(datasets)
    plt.rcParams.update({'font.size': 12})
    ax.figure.tight_layout()
    for dataset in datasets:
        ax.figure.savefig(f'./{dataset.lower()}/output/plots/ct_dist/eval.png', dpi=300)
        df.to_csv(f'./{dataset.lower()}/output/csvs/ct_dist/eval.csv')

def main():
    d = {
        'Accuracy': [],
        'BLEU': [],
        'METEOR': [],
        'chrF++': [],
        'BERT-SCORE F1': [],
        'ROUGE-L F1': [],
    }

    decimal_keys = ['METEOR', 'chrF++', 'ROUGE-L F1']
    real_keys = ['BLEU', 'BERT-SCORE F1']
    datasets = ['MLB', 'SportSett', 'SumTime', 'Obituary']
    for dataset in datasets:
        # auto_file_name = "neural_t5" if dataset == "MLB" or dataset == 'SportSett' else "neural"
        auto_file_name = "neural"
        auto = json.load(open(f"{dataset.lower()}/eval/jsons/{auto_file_name}.json"))
        for k, v in auto.items():
            if k in real_keys:
                # print(k, d)
                d[k].append(v)
            elif k in decimal_keys:
                d[k].append(v * 100)
        # acc_file_name = "neural_t5" if dataset == 'SportSett' else "neural" #dataset == "MLB" or 
        acc_file_name = "neural"
        accs = json.load(open(f"./{dataset.lower()}/eval/jsons/{acc_file_name}_acc_err.json"))
        total_acc = 0
        for k, v in accs.items():
            total_acc += v
        d['Accuracy'].append(total_acc)

    df = pd.DataFrame(d, index=datasets)
    df1 = df.loc[:, ['Accuracy', 'BLEU', 'chrF++', 'BERT-SCORE F1', 'ROUGE-L F1']]
    df2 = df1.rename(columns={'Accuracy': 'Accuracy', 'BLEU': 'BLEU', 'chrF++': 'chrF++', 'BERT-SCORE F1': 'BERT-SCORE', 'ROUGE-L F1': 'ROUGE-L'})
    save_df_dist_plot(df2)

if __name__ == "__main__":
    main()
