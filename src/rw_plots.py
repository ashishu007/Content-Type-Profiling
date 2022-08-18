# Different plots of RotoWire dataset

import json
import re
import spacy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from clf_utils import MultiLabelClassifier, ContentTypeData


class RWPlotter:
    """
    Class for plotting the results of the classifier or message distribution for RotoWire/SportSett dataset
    """

    def __init__(self, num_samples=np.arange(100, 1050, 200), E_CLASS=True, do_abs=False, dataset_name='sportsett'):
        self.num_samples = num_samples
        self.E_CLASS = E_CLASS
        self.targets = ['B', 'W', 'A', 'E'] if self.E_CLASS else ['B', 'W', 'A']
        self.nlp = spacy.load('en_core_web_sm')
        self.dataset_name = dataset_name
        self.do_abs = do_abs

    def raw_sentence(self, sents):
        return ' '.join([sent.text for sent in sents])

    def spacy_sent_tokenize(self, doc):
        sents = []
        all_sents = []
        valid_stop = False
        for sent in doc.sents:
            sents.append(sent)
            valid_stop = True if sent[-1].text in ['.', '?', '!'] else False
            if valid_stop:
                all_sents.append(self.raw_sentence(sents))
                sents = []
        return all_sents

    def save_df_dist_plot(self, df, plt_type='gold', datasets=None):
        """
        type: 'gold' or 'gens'
        """
        if 'mlb' in plt_type:
            ax = df.plot.bar(figsize=(15,5), rot=0)
        else:
            ax = df.plot.bar(figsize=(7, 5), rot=0)
        plt.rcParams.update({'font.size': 12})
        ax.set_title(f'Distribution of Content Type')
        for p in ax.patches:
            ax.annotate(f'{str(int(p.get_height()))}', (p.get_x() * 1.005, (p.get_height() * 1.005) + 2))
        ax.set_ylim(0, 110)
        ax.set_xlabel('Content Type')
        ax.set_ylabel('Percentage')
        print(plt_type, plt_type != "gold_no_split")
        if "gold_no_split" not in plt_type:
            print("if")
            ax.set_xticklabels(['Basic', 'Intra-Event', 'Inter-Event'])
        else:
            print("else")
            ax.legend(labels=['Basic', 'Intra-Event', 'Inter-Event'])
        plt.rcParams.update({'font.size': 12})
        ax.figure.tight_layout()
        if datasets is None:
            ax.figure.savefig(f'./{self.dataset_name}/output/plots/ct_dist/ct_dist_{plt_type}.png', dpi=300)
            df.to_csv(f'./{self.dataset_name}/output/csvs/ct_dist/ct_dist_{plt_type}.csv')
        else:
            for dataset in datasets:
                ax.figure.savefig(f'./{dataset.lower()}/output/plots/ct_dist/ct_dist_{plt_type}.png', dpi=300)
                df.to_csv(f'./{dataset.lower()}/output/csvs/ct_dist/ct_dist_{plt_type}.csv')

    def plot_gold_ct_dist_by_author(self, authors=['Ben Miller', 'Juan Pablo Aravena']):#, 'Nick Brazzoni']):
        auth_dict = {}
        js = [json.loads(i.strip()) for i in open('./sportsett/data/initial/fg_by/only_authors.jsonl', 'r').readlines()]
        authors = set([i['author'] for i in js])
        print(authors, len(authors))
        for auth in authors:
            data = ContentTypeData(target_names=self.targets, path=f'data/initial/fg_by', do_abs=self.do_abs, dataset_name=self.dataset_name)
            summaries = data.get_rw_summaries(task_type='auth', auth=auth)
            clf = MultiLabelClassifier(model_name='bert', ftr_name='none', dataset_name=self.dataset_name, num_classes=len(self.targets))
            messages = []
            for summary in tqdm(summaries):
                doc = self.nlp(summary)
                messages.extend(self.spacy_sent_tokenize(doc))
            print(f"\n\n{auth} has total {len(messages)} messages\n\n")
            preds = clf.predict_multilabel_classif(messages)
            dists = {}
            for idx, label in enumerate(self.targets):
                dists[label] = (np.sum(preds[:, idx])/len(preds))*100
            print(dists)
            auth_dict[auth] = dists
        df = pd.DataFrame(auth_dict)
        print(df)
        self.save_df_dist_plot(df, plt_type='gold_by_author')

    def plot_gold_ct_dist_no_split(self, data_type_for_year_jsons='fg'):
        data = ContentTypeData(target_names=self.targets, path=f'data/initial/fg_by', do_abs=self.do_abs, dataset_name=self.dataset_name)
        summaries = data.get_rw_summaries(task_type='ns', data_type=data_type_for_year_jsons)
        clf = MultiLabelClassifier(model_name='bert', ftr_name='none', dataset_name=self.dataset_name, num_classes=len(self.targets))
        messages = []
        for summary in tqdm(summaries):
            doc = self.nlp(summary)
            messages.extend(self.spacy_sent_tokenize(doc))
        print(f"\n\ntotal {len(messages)} messages\n\n")
        preds = clf.predict_multilabel_classif(messages)
        dists = {}
        for idx, label in enumerate(self.targets):
            dists[label] = (np.sum(preds[:, idx])/len(preds))*100
        print(dists)
        df = pd.DataFrame(dists, index=['GOLD'])
        print(df)
        self.save_df_dist_plot(df, plt_type=f'gold_no_split_{data_type_for_year_jsons}')

    def plot_gold_ct_dist_by_year(self, years=[14, 15, 16, 17, 18], data_type_for_year_jsons='fg'):
        year_dict = {}
        for year in years:
            print(f"\nThis is year: {year}\n")
            data = ContentTypeData(target_names=self.targets, path=f'data/initial/csvs', do_abs=self.do_abs, dataset_name=self.dataset_name)
            summaries = data.get_rw_summaries(task_type='by', year=year, data_type=data_type_for_year_jsons)
            print(f"\n\n{year} has total {len(summaries)} summaries\n\n")
            if data_type_for_year_jsons == 'mlb':
                messages = summaries
            else:
                messages = []
                for summary in tqdm(summaries):
                    doc = self.nlp(summary)
                    messages.extend(self.spacy_sent_tokenize(doc))
            print(f"\nPredicting summaries' content type now...\n")
            clf = MultiLabelClassifier(model_name='bert', ftr_name='none', dataset_name=self.dataset_name, num_classes=len(self.targets))
            preds = clf.predict_multilabel_classif(messages)
            print(f"\nPredicted!\n")
            dists = {}
            for idx, label in enumerate(self.targets):
                dists[label] = (np.sum(preds[:, idx])/len(preds))*100
            print(year, dists)
            year_dict[f'{year}'] = dists
        df = pd.DataFrame(year_dict)
        print(df)
        self.save_df_dist_plot(df, plt_type=f'gold_by_year_{data_type_for_year_jsons}')


def main(plot_type='fg_by', E_CLASS=True, dataset_name='sportsett'):
    do_abs = False if dataset_name == 'sumtime' or 'mlb' in dataset_name else True
    plotter = RWPlotter(E_CLASS=E_CLASS, dataset_name=dataset_name, do_abs=do_abs)
    print(f'\nTargets: {plotter.targets}\t\tDO_ABS: {plotter.do_abs}\n')
    if plot_type == 'ss_by':
        # only for sportsett
        plotter.plot_gold_ct_dist_by_year(data_type_for_year_jsons='ss')
    elif plot_type == 'fg_by':
        # only for sportsett
        plotter.plot_gold_ct_dist_by_year(data_type_for_year_jsons='fg')
    elif plot_type == 'mlb_by':
        # only for mlb
        plotter.plot_gold_ct_dist_by_year(data_type_for_year_jsons='mlb', years=list(range(8, 19)))
    elif plot_type == 'ss_ns':
        # only for rotowire fg 
        # this is gold_ns for rotowire fg
        plotter.plot_gold_ct_dist_no_split(data_type_for_year_jsons='ss')
    elif plot_type == 'fg_ns':
        # only for rotowire fg 
        # this is gold_ns for rotowire fg
        plotter.plot_gold_ct_dist_no_split(data_type_for_year_jsons='fg')
    elif plot_type == 'auth':
        # only for sportsett --> rw_fg
        plotter.plot_gold_ct_dist_by_author()
    else:
        raise ValueError(f'{plot_type} is not a valid argument')


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-type", "--type", help="type of plotting: clf res or content type dist", \
                            default='clf', choices=['ss_by', 'auth', 'fg_by', 'fg_ns', 'ss_ns', 'mlb_by'])
    argParser.add_argument("-e_class", "--e_class", help="plot the performance with A class", action='store_true')
    argParser.add_argument("-dataset", "--dataset", help="dataset name", default='sportsett', \
                            choices=['sportsett', 'obituary', 'sumtime', 'mlb'])

    args = argParser.parse_args()
    print(args)
    main(plot_type=args.type, E_CLASS=args.e_class, dataset_name=args.dataset)

