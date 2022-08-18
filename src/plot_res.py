from operator import index
import spacy
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from clf_utils import MultiLabelClassifier, ContentTypeData

class Plotter:
    """
    Class for plotting the results of the classifier or message distribution
    """

    def __init__(self, num_samples=np.arange(200, 1010, 200), targets=['B', 'W', 'A'], \
                    do_abs=False, dataset_name='sportsett', is_2_class=False):
        self.num_samples = num_samples
        self.targets = targets 
        self.nlp = spacy.load('en_core_web_sm')
        self.dataset_name = dataset_name
        self.do_abs = do_abs
        self.is_2_class = is_2_class

    def raw_sentence(self, sents):
        return ' '.join([sent.text for sent in sents])

    def spacy_sent_tokenize(self, doc):
        # print(f'{doc.text}')
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

    def sumtime_sent_tokenize(self, doc):
        words = doc.strip().split(' ')
        lows = [idx for idx, i in enumerate(words) if i != '' and i[0].islower()]
        sent_lists = [words[i:j] for i, j in zip(lows[:-1], lows[1:])]
        sent_lists.append(words[lows[-1]:])
        sents = [' '.join(i) for i in sent_lists]
        return sents

    def save_df_dist_plot(self, df, plt_type='gold', datasets=None):
        """
        type: 
            'gold', 'gens', 'gold_no_split', 'gold_by_datasets'
            'gold_finer', 'gold_no_split_finer', 'gens_finer', 'gold_by_datasets_finer'
            'gold_by_datasets_2_class', 'gold_no_split_2_class', 'gens_2_class'
        """
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
            if 'finer' in plt_type:
                print("finer")
                ax.set_xticklabels(['Intra-Event Basic', 'Intra-Event Complex', 'Inter-Event', 'External'])
            elif '2_class' in plt_type:
                print('2_class')
                ax.set_xticklabels(['Intra-Event', 'Inter-Event'])
            else:
                print('else')
                ax.set_xticklabels(['Intra-Event Basic', 'Intra-Event Complex', 'Inter-Event'])
        else:
            print("else")
            if 'finer' in plt_type:
                ax.legend(labels=['Intra-Event Basic', 'Intra-Event Complex', 'Inter-Event', 'External'])
            elif '2_class' in plt_type:
                ax.legend(labels=['Intra-Event', 'Inter-Event'])
            else:
                ax.legend(labels=['Intra-Event Basic', 'Intra-Event Complex', 'Inter-Event'])

        plt.rcParams.update({'font.size': 12})
        ax.figure.tight_layout()
        if datasets is None:
            ax.figure.savefig(f'./{self.dataset_name}/output/plots/ct_dist/ct_dist_{plt_type}.png', dpi=300)
            df.to_csv(f'./{self.dataset_name}/output/csvs/ct_dist/ct_dist_{plt_type}.csv')
        else:
            for dataset in datasets:
                # ax.figure.savefig(f'./{dataset.lower()}/output/plots/ct_dist/ct_dist_{plt_type}.png', dpi=300)
                df.to_csv(f'./{dataset.lower()}/output/csvs/ct_dist/ct_dist_{plt_type}.csv')

    def save_df_clf_res_plot(self, df, metric='Accuracy', save_path='./output/plots/clf_res', \
                            is_al_vs_no_al=False, title=None):
        plt.rcParams.update({'font.size': 16})
        ax = df.plot.line(figsize=(10, 5))
        ax.set_xlabel('Number of training samples')
        ax.set_ylabel(f'{metric}')
        ax.set_xticks(self.num_samples)
        title_name = f'{metric} vs Number of training samples' if title is None else title
        ax.set_title(title_name)
        ax.legend(loc='lower right')
        out_file_name_plot = f'./{self.dataset_name}/output/plots/clf_res/{metric.lower()}' \
            if not is_al_vs_no_al else f'./{self.dataset_name}/output/plots/clf_res/{(metric.lower()).replace(" ", "_")}_avna'
        plt.rcParams.update({'font.size': 12})
        ax.figure.tight_layout()
        ax.figure.savefig(f'{out_file_name_plot}.png', dpi=300)
        out_file_name_csv = f'./{self.dataset_name}/output/csvs/clf_res/{metric.lower()}' \
            if not is_al_vs_no_al else f'./{self.dataset_name}/output/csvs/clf_res/{(metric.lower()).replace(" ", "_")}_avna'
        df.to_csv(f'{out_file_name_csv}.csv', index=0)

    def plot_dist_gold_v_clf(self):
        data = ContentTypeData(target_names=self.targets, path=f'./data/tsvs', do_abs=self.do_abs, dataset_name=self.dataset_name)
        test_x, test_y = data.get_data(train=False)
        train_x, train_y = data.get_data(train_file_name='train')
        # all_x, all_y = np.concatenate((train_x, test_x), axis=0), np.concatenate((train_y, test_y), axis=0)
        all_x, all_y = test_x, test_y
        print(f'\n\nall_x.shape: {all_x.shape}\tall_y.shape: {all_y.shape}\n\n')
        clf = MultiLabelClassifier(model_name='bert', ftr_name='none', dataset_name=self.dataset_name, num_classes=len(self.targets))
        pred_y = clf.predict_multilabel_classif(all_x)
        dists = {}
        # print(test_y, pred_y)
        dists['Predicted'] = {label: (np.sum(pred_y[:, idx])/len(pred_y))*100 for idx, label in enumerate(self.targets)}
        dists['Actual'] = {label: (np.sum(np.array(all_y)[:, idx])/len(all_y))*100 for idx, label in enumerate(self.targets)}
        df = pd.DataFrame(dists)
        if len(self.targets) == 2:
            df.loc['A'] = [0.0, 0.0]
        elif len(self.targets) == 1:
            df.loc['W'] = [0.0, 0.0]
            df.loc['A'] = [0.0, 0.0]
        print(df)
        self.save_df_dist_plot(df, plt_type='gold_v_clf')

    def get_2_class_df(self, preds):
        actr, onlyb, onlyw, bothbw = 0, 0, 0, 0
        for item in preds:
            if item[0] == 1 and item[1] == 0:
                onlyb += 1
            elif item[0] == 0 and item[1] == 1:
                onlyw += 1
            elif item[0] == 1 and item[1] == 1:
                bothbw += 1
            if preds.shape[1] == 3:
                if item[2] == 1:
                    actr += 1
        wctr = onlyb + onlyw + bothbw
        return pd.DataFrame({'W': (wctr/len(preds))*100, 'A': (actr/len(preds))*100}, index=['GOLD'])

    def plot_gold_ct_dist_no_split(self):
        all_sents_path = f'data/initial/csvs'
        data = ContentTypeData(target_names=self.targets, path=f'{all_sents_path}', do_abs=self.do_abs, dataset_name=self.dataset_name)
        messages = data.get_only_messages(part='all')
        MODEL_NAME = 'bert' if 'sportsett' in self.dataset_name or 'mlb' in self.dataset_name else 'svm'
        clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name='bert_emb', dataset_name=self.dataset_name, num_classes=len(self.targets))
        if 'finer' in self.dataset_name:
            plt_type = 'gold_no_split_finer'
        elif self.is_2_class:
            plt_type = 'gold_no_split_2_class'
        else:
            plt_type = 'gold_no_split'
        if self.dataset_name != 'obituary':
            preds = clf.predict_multilabel_classif(messages)
        elif self.dataset_name == 'obituary':
            dfs = []
            for part in ['train', 'valid']:
                df = pd.read_csv(f'./{self.dataset_name}/data/tsvs/{part}.tsv', sep='\t')
                dfs.append(df)
            df = pd.concat(dfs)
            preds = np.array(df[self.targets].values.tolist())
        dists = {}
        for idx, label in enumerate(self.targets):
            dists[label] = (np.sum(preds[:, idx])/len(preds))*100
        print(f'\nThis is dists dictionary:\t{dists}\n')
        df = pd.DataFrame(dists, index=['GOLD'])
        if self.is_2_class:
            df = self.get_2_class_df(preds)
        else:
            if len(self.targets) == 2:
                df['A'] = 0.0
            elif len(self.targets) == 1:
                df['W'] = 0.0
                df['A'] = 0.0
        print(df)
        self.save_df_dist_plot(df, plt_type=plt_type)

    def plot_gold_ct_dist_by_datasets(self, is_finer=False):
        datasets = ['MLB', 'SportSett', 'SumTime', 'Obituary'] if not is_finer else ['Finer-SportSett', 'Finer-MLB']
        # plt_type_end = 'gold_by_datasets' if not is_finer else 'gold_by_datasets_finer'
        if is_finer:
            plt_type_end = '_finer'
        elif self.is_2_class:
            plt_type_end = '_2_class'
        else:
            plt_type_end = ''
        plt_type = f'gold_by_datasets{plt_type_end}'
        dfs = []
        for dataset in datasets:
            dfs.append(pd.read_csv(f'./{dataset.lower()}/output/csvs/ct_dist/ct_dist_gold_no_split{plt_type_end}.csv', index_col=0))
        df = pd.concat(dfs, ignore_index=True)
        df.index = datasets
        print(df.transpose())
        self.save_df_dist_plot(df.transpose(), plt_type=plt_type, datasets=datasets)

    def plot_gold_ct_dist_train_test_val_split(self, parts=['train', 'valid', 'test']):
        part_dict = {}
        for part in parts:
            try:
                data = ContentTypeData(target_names=self.targets, path=f'data/initial/csvs', do_abs=self.do_abs, dataset_name=self.dataset_name)
                messages = data.get_only_messages(part=part)
                MODEL_NAME = 'bert' if self.dataset_name == 'sportsett' or self.dataset_name == 'mlb' else 'svm'
                clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name='tfidf', dataset_name=self.dataset_name, num_classes=len(self.targets))
                preds = clf.predict_multilabel_classif(messages)
                dists = {}
                for idx, label in enumerate(self.targets):
                    dists[label] = (np.sum(preds[:, idx])/len(preds))*100
                print(dists)
                part_dict[part.capitalize()] = dists
            except:
                print(f'\n{part} not in {self.dataset_name}\n')
        df = pd.DataFrame(part_dict)
        print(df)
        self.save_df_dist_plot(df, plt_type='gold')

    def plot_gens_ct_dist(self, systems=['rule', 'neural', 'cbr', 'gold']):
        # systems = ['gold_14_test', 'neural_14_test']
        # systems = ['neural', 'gold']

        if self.dataset_name == 'sportsett':
            systems = ['Neural_Ent', 'Neural_MP', 'Neural_Hir', 'Gold']
        elif self.dataset_name == 'mlb':
            systems = ['Neural_Ent', 'Neural_MP', 'Neural_ED', 'Gold']
        elif self.dataset_name == 'sumtime':
            systems = ['Neural_T5', 'Neural_BART', 'Neural_BlTfm', 'Gold']
        elif self.dataset_name == 'obituary':
            systems = ['Neural_T5', 'Neural_BART', 'Neural_BlTfm', 'Gold']
        print(f"\nThis is systems list:\t{systems} for dataset: {self.dataset_name}\n")

        system_dict = {}
        for sys in systems:
            print(f'\nThis is {sys} for {self.dataset_name}\n')
            # try:
            sys_summaries = open(f'./{self.dataset_name}/eval/sys_gens/{sys.lower()}.txt').readlines()
            print(f"\n{sys} found for {self.dataset_name}\n")
            sys_summaries = [x.strip() for x in sys_summaries]
            sys_sents = []
            # if self.dataset_name == 'sportsett' or self.dataset_name == 'obituary' or self.dataset_name == 'mlb':
            if 'sportsett' in self.dataset_name or 'obituary' in self.dataset_name or 'mlb' in self.dataset_name:
                print(f'\nNow tokenising summaries into sentences...\n')
                for summary in tqdm(sys_summaries):
                    doc = self.nlp(summary)
                    sys_sents.extend(self.spacy_sent_tokenize(doc))
                print(f'\nTokenisation done!!\n')
            elif self.dataset_name == 'sumtime':
                for summary in sys_summaries:
                    sys_sents.extend(self.sumtime_sent_tokenize(summary))
            # print(len(sys_sents), set([len(i) for i in sys_sents]))
            if self.do_abs:
                data = ContentTypeData()
                sys_sents = data.abstract_sents(sys_sents)
            MODEL_NAME = 'bert' if self.dataset_name == 'sportsett' or self.dataset_name == 'mlb' else 'svm'
            clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name='tfidf', dataset_name=self.dataset_name, num_classes=len(self.targets))
            preds = clf.predict_multilabel_classif(sys_sents)
            dists = {}
            for idx, label in enumerate(self.targets):
                dists[label] = (np.sum(preds[:, idx])/len(preds))*100
            print(dists)
            system_dict[sys.upper()] = dists
            # except:
            #     print(f'\n{sys} not found for {self.dataset_name}\n')
        df = pd.DataFrame(system_dict)
        print(df)
        if len(self.targets) == 2:
            df.loc['A'] = [0.0] * df.shape[1]
        elif len(self.targets) == 1:
            df.loc['W'] = [0.0] * df.shape[1]
            df.loc['A'] = [0.0] * df.shape[1]
        print(df)
        plt_type = 'finer-gens' if 'finer' in self.dataset_name else 'gens'
        self.save_df_dist_plot(df, plt_type=plt_type)

    def plot_clf_res(self):
        """
        Plot the performance of the classifier
        """
        NUM_SAMPLES = list(self.num_samples)
        MODEL_NAMES = ['bert']#['lr', 'rf', 'svm', 'bert']
        FTR_NAMES = ['none']#['tfidf', 'tf', 'bert_emb']
        # data = ContentTypeData(target_names=self.targets, do_abs=self.do_abs, dataset_name=self.dataset_name)
        data = ContentTypeData(target_names=self.targets, do_abs=False, dataset_name=self.dataset_name)
        accs, mf1s = [], []
        for NUM_SAMPLE in NUM_SAMPLES:
            print(f'NUM_SAMPLEs: {NUM_SAMPLE}')
            train_x, train_y = data.get_data(train=True, num_samples=NUM_SAMPLE)
            test_x, test_y = data.get_data(train=False)
            bert_trained_flag = False
            a, m = {}, {}
            for MODEL_NAME in MODEL_NAMES:
                for FTR_NAME in FTR_NAMES:
                    if not bert_trained_flag:
                        FTR_NAME = 'none' if MODEL_NAME == 'bert' else FTR_NAME
                        config=f'{MODEL_NAME} w/ {FTR_NAME}'
                        print(f'{config}\ton\t{NUM_SAMPLE} training samples')
                        clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name=FTR_NAME, dataset_name=self.dataset_name, num_classes=len(self.targets))
                        clf.train_multilabel_classif(train_x, train_y)
                        bert_trained_flag = True if MODEL_NAME == 'bert' else False
                        FTR_NAME = 'none' if MODEL_NAME == 'bert' else FTR_NAME
                        pred_y = clf.predict_multilabel_classif(test_x)
                        acc, mf1 = clf.report_score(test_y, pred_y, self.targets, 
                                                    num_train_samples=len(train_x), 
                                                    config=config, save_out=True)
                        a[config], m[config] = acc, mf1
                        print(f'\n\n{MODEL_NAME.upper()} with {FTR_NAME.upper()} on {NUM_SAMPLE} samples achieved {acc:.2f}% Accuracy and {mf1:.2f}% Macro F1 score.\n\n')
            accs.append(a)
            mf1s.append(m)
        dfa = pd.DataFrame(accs, index=NUM_SAMPLES)
        dfm = pd.DataFrame(mf1s, index=NUM_SAMPLES)
        self.save_df_clf_res_plot(dfa, metric='Accuracy')
        self.save_df_clf_res_plot(dfm, metric='MacroF1')

    def plot_al_vs_no_al(self):
        print('Plotting AL vs No AL')
        NUM_SAMPLES = list(self.num_samples)
        MODEL_NAME, FTR_NAME = 'bert', 'none'
        data = ContentTypeData(target_names=self.targets, path=f'./data/tsvs', do_abs=self.do_abs, dataset_name=self.dataset_name)
        accs, mf1s = [], []
        for NUM_SAMPLE in NUM_SAMPLES:
            print(f'NUM_SAMPLES: {NUM_SAMPLE}')
            a, m = {}, {}
            for exp_type in ['no_al', 'al']:
                print(f'Expe Type: {exp_type}')
                train_x, train_y = data.get_data(train=True, num_samples=NUM_SAMPLE, train_file_name=f'train_{exp_type}')
                test_x, test_y = data.get_data(train=False)
                # config=f'{MODEL_NAME.upper()} w/ RoBERTa w/ {exp_type.replace("_", " ").upper()}'
                config=f'RobFT w/ {exp_type.replace("_", " ").upper()}'
                print(f'{config}\ton\t{NUM_SAMPLE} training samples for\t{exp_type}')
                clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name=FTR_NAME, dataset_name=self.dataset_name, num_classes=len(self.targets))
                clf.train_multilabel_classif(train_x, train_y)
                pred_y = clf.predict_multilabel_classif(test_x)
                acc, mf1 = clf.report_score(test_y, pred_y, self.targets, 
                                            num_train_samples=len(train_x), 
                                            config=config, save_out=True)
                a[f'Acc for {config}'], m[f'MF1 for {config}'] = acc, mf1
                print(f'Accuracy: {acc}\tMF1: {mf1}')
            accs.append(a)
            mf1s.append(m)
        dfa = pd.DataFrame(accs, index=NUM_SAMPLES)
        dfm = pd.DataFrame(mf1s, index=NUM_SAMPLES)
        df = pd.concat([dfa, dfm], axis=1)
        self.save_df_clf_res_plot(df, metric='Accuracy and MacroF1', \
                                is_al_vs_no_al=True, \
                                title='Accuracy and MacroF1 vs Number of Training Samples')
        # self.save_df_clf_res_plot(dfa, metric='Accuracy', is_al_vs_no_al=True)
        # self.save_df_clf_res_plot(dfm, metric='MacroF1', is_al_vs_no_al=True)


def main(plot_type='gens', targets=['B', 'W', 'A'], dataset_name='sportsett', is_2_class=False):
    do_abs = False if dataset_name == 'sumtime' or 'mlb' in dataset_name else True 
    is_finer = True if 'finer' in dataset_name else False
    plotter = Plotter(targets=targets, dataset_name=dataset_name, do_abs=do_abs, is_2_class=is_2_class)
    print(f'\n{plotter.targets}\t{plotter.do_abs}\n')
    if plot_type == 'clf':
        # only for al in sportsett
        plotter.plot_clf_res()
    elif plot_type == 'gold_bd':
        # for all datasets
        plotter.plot_gold_ct_dist_by_datasets(is_finer=is_finer)
    elif plot_type == 'gold_ttvs':
        # for all datasets --> not neccesary
        plotter.plot_gold_ct_dist_train_test_val_split()
    elif plot_type == 'gold_ns':
        # for all datasets
        plotter.plot_gold_ct_dist_no_split()
    elif plot_type == 'gens':
        # for all datasets
        plotter.plot_gens_ct_dist()
    elif plot_type == 'gvc':
        # only for classifiers --> sportsett and sumtime
        plotter.plot_dist_gold_v_clf()
    elif plot_type == 'avna':
        # only for al v no al --> only for sportsett
        plotter.plot_al_vs_no_al()
    else:
        raise ValueError(f'{plot_type} is not a valid argument')


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-type", "--type", help="type of plotting: clf res or content type dist", \
                            default='clf', choices=['clf', 'gold_ttvs', 'gens', 'gvc', 'avna', 'gold_ns', 'gold_bd']) # avna = al vs no al
    argParser.add_argument("-is_2_class", "--is_2_class", help="plot the performance with only 2 classes (intra and inter event)", action='store_true')
    argParser.add_argument("-a_class", "--a_class", help="plot the performance with A class (across/inter event)", action='store_true')
    argParser.add_argument("-e_class", "--e_class", help="plot the performance with E class (external class)", action='store_true')
    argParser.add_argument("-dataset", "--dataset", help="dataset name", default='sportsett', \
                            choices=['sportsett', 'obituary', 'sumtime', 'mlb', 'finer-sportsett', 'finer-mlb'])

    args = argParser.parse_args()
    print(args)

    targets = ['B', 'W']
    if args.a_class:
        targets.append('A')
    if args.e_class:
        targets.append('E')

    print(f'Targets: {targets}')

    main(plot_type=args.type, targets=targets, dataset_name=args.dataset, is_2_class=args.is_2_class)

