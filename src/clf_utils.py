"""
Supporting functions for classifiers.
"""

import json
import pickle, re, spacy
import pandas as pd
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

import bert_utils as bu

class ContentTypeData:
    """
    train and test data 
    """

    def __init__(self, target_names=['B', 'W', 'A', 'H'], do_abs=True, path='./data/tsvs', dataset_name='sportsett', is_finer=False):
        self.path = path
        self.target_names = target_names
        self.do_abs = do_abs
        self.is_finer = is_finer
        self.dataset_name = dataset_name
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(self.nlp.create_pipe('merge_entities'))

    def abstract_sents(self, messages):
        """
        abstract sentences using spacy ner tagger
        """
        print(f'\nInside ContentTypeData.abstract_sents() method\n')
        abs_messages = []
        for message in tqdm(messages):
            if self.dataset_name == 'sportsett':
                message = re.sub('\(\d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}\)', '(ShotBreakdown)', message)
                message = re.sub('\s+', ' ', message)
            doc = self.nlp(message)
            abs_messages.append(" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc]))
        print(f'\nGoing out of ContentTypeData.abstract_sents() method\n')
        return abs_messages

    def get_data(self, train=True, num_samples=0, train_file_name=None):
        """
        Get data for training and testing multi-label content type classifiers.
        Can be used for any dataset.
        param 'train_file_name' is used for sportsett dataset when experimenting with al vs no al
        """
        train_file = f'./{self.dataset_name}/{self.path}/train.tsv' if train_file_name is None else \
            f'./{self.dataset_name}/{self.path}/{train_file_name}.tsv'
        df1 = pd.read_csv(train_file, delimiter='\t') if train else \
            pd.read_csv(f'./{self.dataset_name}/{self.path}/valid.tsv', delimiter='\t')

        if self.is_finer:
            print("\nis finer\n")
            ig_inds = [i for i, row in df1.iterrows() if row['B'] == 0 and row['W'] == 0 and row['A'] == 0 and row['E'] == 0]
        else:
            print("\nis not finer\n")
            ig_inds = [i for i, row in df1.iterrows() if row['B'] == 0 and row['W'] == 0 and row['A'] == 0]

        bad_df = df1.index.isin(ig_inds)
        df = df1[~bad_df]
        messages_from_tsv = list(df['Sentence'])
        num_samples = len(messages_from_tsv) if num_samples == 0 else num_samples
        messages_from_tsv = messages_from_tsv[:num_samples]
        print(f'You are getting {num_samples} samples for train == {train}')
        messages = self.abstract_sents(messages_from_tsv) if self.do_abs else messages_from_tsv
        labels = df[self.target_names].values.tolist()[:num_samples]
        return np.array(messages), np.array(labels)

    def get_unlabelled_data_for_al(self, num_samples=100000):
        raw_messages = open(f'./{self.dataset_name}/data/txts/unlabelled.txt', 'r').readlines()
        num_samples = len(raw_messages) if num_samples == 0 else num_samples
        unlabelled_x = [x.strip() for x in raw_messages][:num_samples]
        print(f'\nYou are getting {len(unlabelled_x)} samples from unlabelled pool\n')
        unlabelled_x_abs = self.abstract_sents(unlabelled_x) if self.do_abs else unlabelled_x
        return unlabelled_x, unlabelled_x_abs

    def get_only_messages(self, part='train', num_samples=0):
        """
        this is used for getting only messages from the different splits (train/test/valid) of the dataset
        """
        if part == 'all':
            dfs = []
            parts = ['train', 'test', 'valid']
            for p in parts:
                print(f'\nReading {p} data from {self.dataset_name} for get_only_messages\n')
                try:
                    df = pd.read_csv(f'./{self.dataset_name}/{self.path}/{p}.csv')
                    dfs.append(df)# if 'mlb' not in self.dataset_name else dfs.append(df.iloc[:100000, :])
                except:
                    print(f'\n{p} not found in {self.dataset_name}\n')
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(f'./{self.dataset_name}/{self.path}/{part}.csv')
        messages = df['raw'].values.tolist()
        print(f'\nabstracting sentences\n')
        messages = self.abstract_sents(messages) if self.do_abs else messages
        num_samples = len(messages) if num_samples == 0 else num_samples
        print(f"\nYou are getting {num_samples} samples\n")
        return messages[:num_samples]

    def get_rw_summaries(self, task_type='auth', author_name='Ben Miller', data_type='fg', year=14, num_samples=0):
        """
        used for getting different data types for sportsett/rotowire summaries
        task_type: 
            auth --> get summary by authors
                if task_type == 'auth' --> must provide author_name parameter
            ns --> get ss and fg summaries no split
                if task_type == 'ns' --> must provide datat_type parameter
            by --> get summaries by year
                if task_type == 'by' --> must provide data_type and year parameter
                data_type: fg, ss --> fg is for rotowire_fg, ss is for sportsett
        """
        if task_type == 'auth':
            summaries = []
            js = [json.loads(i.strip()) for i in open(f'./{self.dataset_name}/data/initial/fg_by/only_authors.jsonl', 'r').readlines()]
            summaries.extend([item['summary'] for item in js if author_name.lower() == item['author'].lower()])
        elif task_type == 'ns':
            summaries = []
            for season in [14, 15, 16, 17, 18]:
                year_json_path_name = f'./{self.dataset_name}/data/initial/fg_by/{season}.json' if data_type == 'fg' \
                    else f'./{self.dataset_name}/data/initial/jsons/{season}.json'
                js = json.load(open(year_json_path_name))
                summaries.extend([' '.join(item['summary']) for item in js])
        elif task_type == 'by':
            if data_type == 'mlb':
                dfs = pd.concat([pd.read_csv(f'./{self.dataset_name}/data/initial/csvs/{part}.csv') for part in ['train', 'test', 'valid']])
                summaries = dfs[dfs['season'] == year]['raw'].values.tolist()
            else:
                year_json_path_name = f'./{self.dataset_name}/data/initial/fg_by/{year}.json' if data_type == 'fg' \
                    else f'./{self.dataset_name}/data/initial/jsons/{year}.json'
                js = json.load(open(year_json_path_name))
                summaries = [' '.join(item['summary']) for item in js]
        print(len(summaries))#, summaries[0])
        summaries = self.abstract_sents(summaries) if self.do_abs else summaries
        num_samples = len(summaries) if num_samples == 0 else num_samples
        print(f"\n\nYou are getting {num_samples} summaries\n\n")
        return summaries[:num_samples]


class TextFeatureExtractor:
    """
    type = [tfidf, tf, bert_emb]
    """

    def __init__(self, type='tfidf', path='./output/ftrs', dataset_name='sportsett'):
        self.type = type
        self.path = path
        self.dataset_name = dataset_name
        self.embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def extract_train_features(self, messages):
        if self.type == 'tfidf' or self.type == 'tf':
            vectorizer = TfidfVectorizer() if self.type == 'tfidf' else CountVectorizer()
            vectorizer.fit(messages)
            pickle.dump(vectorizer, open(f'./{self.dataset_name}/{self.path}/{self.type}_vect.pkl', 'wb'))
            return vectorizer.transform(messages)
        elif self.type == 'bert_emb':
            return self.embedding_model.encode(messages)
        else:
            raise ValueError(f'{self.type} is not supported')

    def extract_test_features(self, messages):
        if self.type == 'tfidf' or self.type == 'tf':
            vectorizer = pickle.load(open(f'./{self.dataset_name}/{self.path}/{self.type}_vect.pkl', 'rb'))
            return vectorizer.transform(messages)
        elif self.type == 'bert_emb':
            return self.embedding_model.encode(messages)
        else:
            raise ValueError(f'{self.type} is not supported')


class MultiLabelClassifier:
    """
    model_name = [svm, lr, rf, bert]
    """

    def __init__(self, model_name='svm', ftr_name='tfidf', model_path='./output/models', dataset_name='sportsett', num_classes=3):
        self.path = model_path
        self.model_name = model_name
        self.ftr_name = ftr_name
        self.save_model_name = f'{self.model_name}_w_{self.ftr_name}'
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.ftr_extractor = TextFeatureExtractor(self.ftr_name, dataset_name=self.dataset_name)

    def train_multilabel_classif(self, messages, labels):
        if self.model_name == 'svm' or self.model_name == 'lr' or self.model_name == 'rf':
            if self.model_name == 'svm':
                self.clf = MultiOutputClassifier(SVC(kernel='linear', probability=True, random_state=0))
            elif self.model_name == 'lr':
                self.clf = MultiOutputClassifier(LogisticRegression(random_state=0))
            elif self.model_name == 'rf':
                self.clf = MultiOutputClassifier(RandomForestClassifier(random_state=0))
            self.messages_ftr = self.ftr_extractor.extract_train_features(messages)
            self.clf.fit(self.messages_ftr, labels)
            pickle.dump(self.clf, open(f'./{self.dataset_name}/{self.path}/multilabel_{self.save_model_name}.pkl', 'wb'))
            return self.clf
        elif self.model_name == 'bert':
            num_epochs = 20 if 'mlb' in self.dataset_name else 7
            return bu.train_bert_multilabel_classif(messages, labels, num_epochs=num_epochs, dataset=self.dataset_name, num_classes=self.num_classes, \
                                                    path=f'./{self.dataset_name}/{self.path}/multilabel_roberta.pt')
        else:
            raise ValueError(f'{self.model_name} is not supported')

    def predict_multilabel_classif(self, messages, pred_probs=False):
        if self.model_name == 'svm' or self.model_name == 'lr' or self.model_name == 'rf':
            print(f'Extracting {self.ftr_name} features...')
            self.messages_ftr = self.ftr_extractor.extract_test_features(messages)
            print(f'{self.ftr_name} features extracted!!!')
            self.model = pickle.load(open(f'./{self.dataset_name}/{self.path}/multilabel_{self.save_model_name}.pkl', 'rb'))
            return self.model.predict_proba(self.messages_ftr) if pred_probs else self.model.predict(self.messages_ftr)
        elif self.model_name == 'bert':
            return bu.predict_bert_multilabel_classif(messages, pred_probs=pred_probs, dataset=self.dataset_name, num_classes=self.num_classes, \
                                                    path=f'./{self.dataset_name}/{self.path}/multilabel_roberta.pt')
        else:
            raise ValueError(f'{self.model_name} is not supported')

    def report_score(self, true_y, pred_y, target_names, num_train_samples, config='svm w/ tfidf', save_out=True):
        acc = accuracy_score(true_y, pred_y)*100
        mf1 = f1_score(true_y, pred_y, average='macro')*100
        # print(f'{config}\tAcc: {acc:.2f}\tMF1: {mf1:.2f}')
        # print(classification_report(true_y, pred_y, target_names=target_names))
        if save_out:
            print(f'./{self.dataset_name}/output/report.txt')
            with open(f'./{self.dataset_name}/output/report.txt', 'a') as f:
                f.write(f'{config}\n')
                f.write(f'# Train Samples: {num_train_samples}\n')
                f.write(f'Acc: {acc:.2f}\tMF1: {mf1:.2f}\n')
                f.write(classification_report(true_y, pred_y, target_names=target_names))
                f.write('\n\n')
            f.close()
        return acc, mf1

