"""
Main file
"""

import argparse
import numpy as np
from operator import itemgetter 
from al_utils import ALUtil
from clf_utils import ContentTypeData, MultiLabelClassifier

def main(QUERY_STRATEGY='margin', TARGETS=['B', 'W'], TOP_K=10, DO_AL=True, DATASET='sportsett'):
    DO_ABS = False if DATASET == 'sumtime' or 'mlb' in DATASET else True
    data = ContentTypeData(path=f'./data/tsvs', target_names=TARGETS, do_abs=DO_ABS, dataset_name=DATASET)
    print(f'\n\n{data.target_names}\n\n')
    train_file_name = 'train_al' if DATASET == 'sportsett' else 'train'
    train_x, train_y = data.get_data(train=True, train_file_name=train_file_name, num_samples=610)
    test_x, test_y = data.get_data(train=False)

    MODEL_NAMES = ['lr', 'rf', 'svm', 'bert'] if QUERY_STRATEGY == 'qbc' or QUERY_STRATEGY == 'clust' else ['bert']
    FTR_NAMES = ['tfidf', 'tf', 'bert_emb'] if QUERY_STRATEGY == 'qbc' or QUERY_STRATEGY == 'clust' else ['bert_emb']

    print(f'\n\nModels: {MODEL_NAMES}\tFeatures: {FTR_NAMES}\n\n')

    bert_trained_flag = False # since bert needs to be trained only once
    all_preds = []
    for MODEL_NAME in MODEL_NAMES:
        for FTR_NAME in FTR_NAMES:
            if not bert_trained_flag: # make sure bert is trained only in the end
                clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name=FTR_NAME, dataset_name=DATASET, num_classes=len(TARGETS))
                print(f'\n\n{MODEL_NAME} w/ {FTR_NAME}\tTraining...')
                # Just training the model here
                trained_model = clf.train_multilabel_classif(train_x, train_y)
                bert_trained_flag = True if MODEL_NAME == 'bert' else False
                FTR_NAME = 'none' if MODEL_NAME == 'bert' else FTR_NAME
                print(f'{MODEL_NAME} w/ {FTR_NAME}\tPredicting...')
                pred_y = clf.predict_multilabel_classif(test_x)
                acc, mf1 = clf.report_score(test_y, pred_y, data.target_names, 
                                            num_train_samples=len(train_x), 
                                            config=f'{MODEL_NAME} w/ {FTR_NAME}')
                print(f'{MODEL_NAME} w/ {FTR_NAME}\tAcc: {acc:.2f}\tMF1: {mf1:.2f}\n\n')
    if DO_AL:
        alu = ALUtil(top_k=TOP_K)
        unlabelled_x, unlabelled_x_abs = data.get_unlabelled_data_for_al()

        print('\n\n*******************************************')
        print('Now predicting unlabelled for AL...')
        print('*******************************************\n\n')
        bert_trained_flag = False # since bert needs to be trained only once
        for MODEL_NAME in MODEL_NAMES:
            for FTR_NAME in FTR_NAMES:
                if not bert_trained_flag: # make sure bert is trained only in the end
                    FTR_NAME = 'none' if MODEL_NAME == 'bert' else FTR_NAME
                    print(f'\n\n{MODEL_NAME} w/ {FTR_NAME}\tPredicting unlabelled...')
                    bert_trained_flag = True if MODEL_NAME == 'bert' else False
                    # Now using the models trained previously to predict on unlabelled data
                    clf = MultiLabelClassifier(model_name=MODEL_NAME, ftr_name=FTR_NAME, dataset_name=DATASET)
                    unlabelled_pred_y = clf.predict_multilabel_classif(unlabelled_x_abs, pred_probs=True)
                    unlabelled_pred_y = np.array([i[:, 1] for i in unlabelled_pred_y]).transpose() \
                                    if MODEL_NAME != 'bert' else unlabelled_pred_y
                    all_preds.append(unlabelled_pred_y)
                    print(f'{MODEL_NAME} w/ {FTR_NAME}\tPredicted...\n\n')

        if QUERY_STRATEGY == 'clust':
            clust_cents = np.load(f'./{DATASET}/data/base/clust_cents.npy')
            top_k_messages_indices, remaining_messages_indices = alu.cluster_sampling(all_preds, unlabelled_x, clust_cents)
        elif QUERY_STRATEGY == 'qbc':
            top_k_messages_indices, remaining_messages_indices = alu.qbc_sampling(all_preds)
        elif QUERY_STRATEGY == 'margin':
            print(f'{MODEL_NAME} w/ {FTR_NAME}\tPredicting unlabelled...')
            unlabelled_pred_y = clf.predict_multilabel_classif(unlabelled_x_abs, pred_probs=True)
            unlabelled_pred_y = np.array([i[:, 1] for i in unlabelled_pred_y]).transpose() \
                            if MODEL_NAME != 'bert' else unlabelled_pred_y
            top_k_messages_indices, remaining_messages_indices = alu.margin_sampling(unlabelled_pred_y)

        top_k_messages, remaining_messages = itemgetter(*top_k_messages_indices)(unlabelled_x), \
                                                itemgetter(*remaining_messages_indices)(unlabelled_x)
        # print(top_k_messages_indices)

        # save the top k messages to a file
        with open(f'./{DATASET}/data/txts/top_{TOP_K}_unlabelled.txt', 'w') as f:
            f.write('\n'.join(top_k_messages))
        # save the remaining unlablelled messages from the pool to the same file
        with open(f'./{DATASET}/data/txts/unlabelled.txt', 'w') as f:
            f.write('\n'.join(remaining_messages))


if __name__ == '__main__':
    """
    python3 src/al_main.py --dataset mlb -qs qbc -a_class
    """

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-do_al", "--do_al", help="do AL (means save top_k_unlabelled.txt)", action='store_true')
    argParser.add_argument("-qs", "--qs", help="query strategy", default='margin', choices=['margin', 'qbc', 'clust'])
    argParser.add_argument("-tk", "--tk", help="top k", default=25, type=int)
    argParser.add_argument("-e_class", "--e_class", help="plot the performance with E class (external class)", action='store_true')
    argParser.add_argument("-a_class", "--a_class", help="use A class or not", action='store_true')
    argParser.add_argument("-dataset", "--dataset", help="dataset name", default='sportsett', \
                            choices=['sportsett', 'obituary', 'sumtime', 'mlb', 'finer-mlb', 'finer-sportsett'])

    args = argParser.parse_args()
    print(args)

    targets = ['B', 'W']
    if args.a_class:
        targets.append('A')
    if args.e_class:
        targets.append('E')
    
    print(f'targets: {targets}')

    main(QUERY_STRATEGY=args.qs, TOP_K=args.tk, TARGETS=targets, DO_AL=args.do_al, DATASET=args.dataset)

