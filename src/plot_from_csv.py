import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_name_dict = {
    'mlb'       : 'MLB',
    'sportsett' : 'SportSett',
    'sumtime'   : 'SumTime',
    'obituary'  : 'Obituary'
}

def save_df_dist_plot(df, plt_type='gens', dataset='sportsett', type_str='gens', title=None):
    """
    type: 'gold' or 'gens'
    """

    datasets = ['mlb', 'sportsett', 'sumtime', 'obituary'] if plt_type != 'gens' else [dataset]

    plt.rcParams.update({'font.size': 16})
    ax = df.plot.bar(figsize=(10, 6), rot=0)

    for p in ax.patches:
        ax.annotate(f'{str(int(p.get_height()))}', (p.get_x() * 1.005, (p.get_height() * 1.005) + 2))

    ax.set_ylim(0, 110)
    # ax.set_xlabel('Content Type')
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(['Intra Basic', 'Intra Complex', 'Inter'])
    if title is None:
        title_str = 'Test Gold' if 'gold' in plt_type else 'Test Generated'
        title_str = 'All Gold' if type_str == 'gold_by_datasets' else title_str
        ax.set_title(f'Content Type Distribution in {title_str} Summaries')
    if type_str == 'gold_by_splits':
        ax.set_title(f'Content Type Distribution by Different Splits in {dataset_name_dict[dataset]}')
    ax.figure.tight_layout()
    plt.rcParams.update({'font.size': 16})

    if plt_type == 'gens' or plt_type == 'splits':
        # ax.set_title(f'{datasets[0]}')
        ax.figure.tight_layout()
        plt.rcParams.update({'font.size': 16})
        ax.figure.savefig(f'./{dataset.lower()}/output/plots/ct_dist/ct_dist_{type_str}.png', dpi=300)
    else:
        for dataset_name in datasets:
            print(f'Saving {dataset_name}')
            ax.figure.savefig(f'./{dataset_name.lower()}/output/plots/ct_dist/ct_dist_{type_str}.png', dpi=300)

def test_set_by_datasets(type='gens'):
    """
    type: 'gens' or 'gold'
    """
    diction = {}
    for folder_name, dataset_name in dataset_name_dict.items():
        df = pd.read_csv(f'./{folder_name}//output/csvs/ct_dist/ct_dist_gens.csv')
        diction[dataset_name] = df['NEURAL'].to_list() if type == 'gens' else df['GOLD'].to_list()
    df = pd.DataFrame(diction, index=['B', 'W', 'A'])
    print(df)
    save_df_dist_plot(df, plt_type=f'{type}_test', dataset='sportsett', type_str=f'{type}_by_datasets_test')

def all_set_by_datasets(plt_type='gens', dataset='sportsett'):
    type_str = 'gens' if plt_type == 'gens' else 'gold_by_datasets'
    df = pd.read_csv(f'{dataset}/output/csvs/ct_dist/ct_dist_{type_str}.csv')
    save_df_dist_plot(df, plt_type=plt_type, dataset=dataset_name_dict[dataset], type_str=type_str)

def all_set_by_split(plt_type='gens', dataset='mlb'):
    print('inside all_set_by_split')
    df = pd.read_csv(f'{dataset}/output/csvs/ct_dist/ct_dist_gold.csv')
    save_df_dist_plot(df, plt_type=plt_type, dataset=dataset, type_str="gold_by_splits", title=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', '-type', type=str, default='gens', \
                        help='Type of plot to make', choices=['gens', 'gold', 'splits'])
    parser.add_argument('--set', '-set', type=str, default='all', \
                        help='To plot for test set or all combined', choices=['all', 'test'])
    parser.add_argument('--dataset', '-dataset', type=str, default='sportsett', \
                        choices=['sportsett', 'mlb', 'sumtime', 'obituary'])

    args = parser.parse_args()

    if args.set == 'all':
        if args.type == 'gens':
            all_set_by_datasets(plt_type=args.type, dataset=args.dataset)
        elif args.type == 'splits':
            all_set_by_split(plt_type=args.type, dataset=args.dataset)
    elif args.set == 'test':
        test_set_by_datasets(args.type)

