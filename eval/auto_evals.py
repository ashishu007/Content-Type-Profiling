__author__='thiagocastroferreira'

"""
Author: Organizers of the 2nd WebNLG Challenge
Date: 23/04/2020
Description:
    This script aims to evaluate the output of data-to-text NLG models by 
    computing popular automatic metrics such as BLEU (two implementations), 
    METEOR, chrF++, TER and BERT-Score.
    
    ARGS:
        usage: eval.py [-h] -R REFERENCE -H HYPOTHESIS [-lng LANGUAGE] [-nr NUM_REFS]
               [-m METRICS] [-nc NCORDER] [-nw NWORDER] [-b BETA]

        optional arguments:
          -h, --help            show this help message and exit
          -R REFERENCE, --reference REFERENCE
                                reference translation
          -H HYPOTHESIS, --hypothesis HYPOTHESIS
                                hypothesis translation
          -lng LANGUAGE, --language LANGUAGE
                                evaluated language
          -nr NUM_REFS, --num_refs NUM_REFS
                                number of references
          -m METRICS, --metrics METRICS
                                evaluation metrics to be computed
          -nc NCORDER, --ncorder NCORDER
                                chrF metric: character n-gram order (default=6)
          -nw NWORDER, --nworder NWORDER
                                chrF metric: word n-gram order (default=2)
          -b BETA, --beta BETA  chrF metric: beta parameter (default=2)

    EXAMPLE:
        ENGLISH: 
            python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 4 -m bleu,meteor,chrf++,ter,bert,bleurt
        RUSSIAN:
            python3 eval.py -R data/ru/reference -H data/ru/hypothesis -lng ru -nr 1 -m bleu,meteor,chrf++,ter,bert

    Multiple References:
        In case of multiple references, they have to be stored in separated files and named reference0, reference1, reference2, etc.

"""

import argparse
import codecs
import copy
import os
import pyter
import logging
import nltk
import subprocess
import re
import json
import datasets
import pandas as pd
import numpy as np

from bert_score import score
from metrics.chrF import computeChrF
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from razdel import tokenize
from tabulate import tabulate

BLEU_PATH = 'eval/metrics/multi-bleu-detok.perl'
METEOR_PATH = 'eval/metrics/meteor-1.5/meteor-1.5.jar'


def parse(refs_path, hyps_path, num_refs, lng='en'):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.read().split('\n')
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text])
                else:
                    references[j].append(text)
    # print('references:', len(references))
    # print(references)
    # references tokenized

    references_tok = copy.copy(references)
    for i, refs in enumerate(references_tok):
        if lng == 'ru':
            references_tok[i] = [' '.join([_.text for _ in tokenize(ref)]) for ref in refs]
        else:
            references_tok[i] = [' '.join(nltk.word_tokenize(ref)) for ref in refs]

    # hypothesis
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        hypothesis = f.read().split('\n')

    # hypothesis tokenized
    hypothesis_tok = copy.copy(hypothesis)
    if lng == 'ru':
        hypothesis_tok = [' '.join([_.text for _ in tokenize(hyp)]) for hyp in hypothesis_tok]
    else:
        hypothesis_tok = [' '.join(nltk.word_tokenize(hyp)) for hyp in hypothesis_tok]


    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, references_tok, hypothesis, hypothesis_tok

def bleu_score(refs_path, hyps_path, num_refs):
    logging.info('STARTING TO COMPUTE BLEU...')
    print('STARTING TO COMPUTE BLEU...')
    ref_files = []
    for i in range(num_refs):
        if num_refs == 1:
            ref_files.append(refs_path)
        else:
            ref_files.append(refs_path + str(i))

    command = 'perl {0} {1} < {2}'.format(BLEU_PATH, ' '.join(ref_files), hyps_path)
    result = subprocess.check_output(command, shell=True)
    try:
        bleu = float(re.findall('BLEU = (.+?),', str(result))[0])
    except:
        logging.error('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE PERL INSTALLED GLOBALLY ON YOUR MACHINE.')
        print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE PERL INSTALLED GLOBALLY ON YOUR MACHINE.')
        bleu = -1
    logging.info('FINISHING TO COMPUTE BLEU...')
    print('FINISHING TO COMPUTE BLEU...')
    return bleu


def bleu_nltk(references, hypothesis):
    # check for empty lists
    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append([ref.split() for ref in refs_])
            hypothesis_.append(hypothesis[i].split())

    chencherry = SmoothingFunction()
    return corpus_bleu(references_, hypothesis_, smoothing_function=chencherry.method3)


def meteor_score(references, hypothesis, num_refs, lng='en'):
    logging.info('STARTING TO COMPUTE METEOR...')
    print('STARTING TO COMPUTE METEOR...')
    hyps_tmp, refs_tmp = 'hypothesis_meteor', 'reference_meteor'

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis))

    linear_references = []
    for refs in references:
        for i in range(num_refs):
            linear_references.append(refs[i])

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    try:
        command = 'java -Xmx2G -jar {0} '.format(METEOR_PATH)
        command += '{0} {1} -l {2} -norm -r {3}'.format(hyps_tmp, refs_tmp, lng, num_refs)
        result = subprocess.check_output(command, shell=True)
        meteor = result.split(b'\n')[-2].split()[-1]
    except:
        logging.error('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        print('ERROR ON COMPUTING METEOR. MAKE SURE YOU HAVE JAVA INSTALLED GLOBALLY ON YOUR MACHINE.')
        meteor = -1

    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE METEOR...')
    print('FINISHING TO COMPUTE METEOR...')
    return float(meteor)


def chrF_score(references, hypothesis, num_refs, nworder, ncorder, beta):
    logging.info('STARTING TO COMPUTE CHRF++...')
    print('STARTING TO COMPUTE CHRF++...')
    hyps_tmp, refs_tmp = 'hypothesis_chrF', 'reference_chrF'

    # check for empty lists
    references_, hypothesis_ = [], []
    for i, refs in enumerate(references):
        refs_ = [ref for ref in refs if ref.strip() != '']
        if len(refs_) > 0:
            references_.append(refs_)
            hypothesis_.append(hypothesis[i])

    with codecs.open(hyps_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(hypothesis_))

    linear_references = []
    for refs in references_:
        linear_references.append('*#'.join(refs[:num_refs]))

    with codecs.open(refs_tmp, 'w', 'utf-8') as f:
        f.write('\n'.join(linear_references))

    rtxt = codecs.open(refs_tmp, 'r', 'utf-8')
    htxt = codecs.open(hyps_tmp, 'r', 'utf-8')

    try:
        totalF, averageTotalF, totalPrec, totalRec = computeChrF(rtxt, htxt, nworder, ncorder, beta, None)
    except:
        logging.error('ERROR ON COMPUTING CHRF++.')
        print('ERROR ON COMPUTING CHRF++.')
        totalF, averageTotalF, totalPrec, totalRec = -1, -1, -1, -1
    try:
        os.remove(hyps_tmp)
        os.remove(refs_tmp)
    except:
        pass
    logging.info('FINISHING TO COMPUTE CHRF++...')
    print('FINISHING TO COMPUTE CHRF++...')
    return totalF, averageTotalF, totalPrec, totalRec


def ter_score(references, hypothesis, num_refs):
    logging.info('STARTING TO COMPUTE TER...')
    print('STARTING TO COMPUTE TER...')
    ter_scores = []
    for hyp, refs in zip(hypothesis, references):
        candidates = []
        for ref in refs[:num_refs]:
            if len(ref) == 0:
                ter_score = 1
            else:
                try:
                    ter_score = pyter.ter(hyp.split(), ref.split())
                except:
                    ter_score = 1
            candidates.append(ter_score)

        ter_scores.append(min(candidates))

    logging.info('FINISHING TO COMPUTE TER...')
    print('FINISHING TO COMPUTE TER...')
    return sum(ter_scores) / len(ter_scores)


def run(refs_path, hyps_path, num_refs, lng='en', metrics='bleu,meteor,chrf++,ter,bert,bleurt',ncorder=6, nworder=2, beta=2):
    metrics = metrics.lower().split(',')
    print(f"\n\nMetrics: {metrics}\n\n")
    metrics = [metric.strip() for metric in metrics if metric != '']
    references, references_tok, hypothesis, hypothesis_tok = parse(refs_path, hyps_path, num_refs, lng)
    
    refs_for_hf = open(f"{refs_path}").readlines()
    hyps_for_hf = open(f"{hyps_path}").readlines()

    result = {}
    
    logging.info('STARTING EVALUATION...')
    print('STARTING EVALUATION...')
    if 'bleu' in metrics:
        bleu = bleu_score(refs_path, hyps_path, num_refs)
        result['bleu'] = bleu

        b = bleu_nltk(references_tok, hypothesis_tok)
        result['bleu_nltk'] = b
    if 'meteor' in metrics:
        print('STARTING TO COMPUTE METEOR...')
        meteor_hf = datasets.load_metric("meteor")
        meteor_hf.add_batch(predictions=hyps_for_hf, references=refs_for_hf)
        score = meteor_hf.compute()
        result['meteor-hf'] = score['meteor']
        print('FINISHING TO COMPUTE METEOR...')
    if 'chrf++' in metrics:
        chrf, _, _, _ = chrF_score(references, hypothesis, num_refs, nworder, ncorder, beta)
        result['chrf++'] = chrf
    if 'ter' in metrics:
        ter = ter_score(references_tok, hypothesis_tok, num_refs)
        result['ter'] = ter
    if 'rouge' in metrics:
        print('STARTING TO COMPUTE ROUGE...')
        rouge = datasets.load_metric("rouge")
        rouge.add_batch(predictions=hyps_for_hf, references=refs_for_hf)
        score = rouge.compute(use_agregator=True)
        result['rouge_p'] = score['rougeL'].mid.precision
        result['rouge_r'] = score['rougeL'].mid.recall
        result['rouge_f'] = score['rougeL'].mid.fmeasure
        print('FINISHING TO COMPUTE ROUGE...')
    if 'bert' in metrics:
        print('STARTING TO COMPUTE BERTSCORE...')
        bert_score = datasets.load_metric("bertscore")
        bert_score.add_batch(predictions=hyps_for_hf, references=refs_for_hf)
        score = bert_score.compute(lang='en')
        result['bert_precision'] = round(np.mean(score['precision'])*100, 2)
        result['bert_recall'] = round(np.mean(score['recall'])*100, 2)
        result['bert_f1'] = round(np.mean(score['f1'])*100, 2)
        print('FINISHING TO COMPUTE BERTSCORE...')
    if 'bleurt' in metrics and lng == 'en':
        bleurt_score = datasets.load_metric("bleurt")
        bleurt_score.add_batch(predictions=hyps_for_hf, references=refs_for_hf)
        score = bleurt_score.compute()
        s = np.mean(np.array(score['scores']))
        result['bleurt-hf'] = s

    logging.info('FINISHING EVALUATION...')
    print('FINISHING EVALUATION...')

    return result


if __name__ == '__main__':
    FORMAT = '%(levelname)s: %(asctime)-15s - %(message)s'
    logging.basicConfig(filename='eval.log', level=logging.INFO, format=FORMAT)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-reference", "--reference", help="reference translation", required=True)
    argParser.add_argument("-hypothesis", "--hypothesis", help="hypothesis translation", required=True)
    argParser.add_argument("-language", "--language", help="evaluated language", default='en')
    argParser.add_argument("-num_refs", "--num_refs", help="number of references", type=int, default=1)
    argParser.add_argument("-metrics", "--metrics", help="evaluation metrics to be computed", default='bleu,meteor,chrf++,bert,bleurt')
    argParser.add_argument("-ncorder", "--ncorder", help="chrF metric: character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nworder", "--nworder", help="chrF metric: word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-beta", "--beta", help="chrF metric: beta parameter (default=2)", type=float, default=2.0)

    args = argParser.parse_args()
    print(args)

    logging.info('READING INPUTS...')
    refs_path = args.reference
    hyps_path = args.hypothesis
    lng = args.language
    num_refs = args.num_refs
    metrics = args.metrics

    nworder = args.nworder
    ncorder = args.ncorder
    beta = args.beta
    logging.info('FINISHING TO READ INPUTS...')

    system_name = hyps_path.split('/')[-1].split('.')[0]
    dataset_name = refs_path.split('/')[1]

    result = run(refs_path=refs_path, hyps_path=hyps_path, num_refs=num_refs, lng=lng, metrics=metrics, ncorder=ncorder, nworder=nworder, beta=beta)
    
    metrics = metrics.lower().split(',')

    headers, values = [], []
    if 'bleu' in metrics:
        headers.append('BLEU')
        values.append(result['bleu'])

        headers.append('BLEU NLTK')
        values.append(round(result['bleu_nltk'], 2))
    if 'meteor' in metrics:
        headers.append('METEOR')
        values.append(round(result['meteor-hf'], 2))
    if 'chrf++' in metrics:
        headers.append('chrF++')
        values.append(round(result['chrf++'], 2))
    if 'ter' in metrics:
        headers.append('TER')
        values.append(round(result['ter'], 2))
    if 'bert' in metrics:
        headers.append('BERT-SCORE P')
        values.append(round(result['bert_precision'], 2))
        headers.append('BERT-SCORE R')
        values.append(round(result['bert_recall'], 2))
        headers.append('BERT-SCORE F1')
        values.append(round(result['bert_f1'], 2))
    if 'bleurt' in metrics and lng == 'en':
        headers.append('BLEURT')
        values.append(round(result['bleurt-hf'], 2))
    if 'rouge' in metrics:
        headers.append('ROUGE-L P')
        values.append(round(result['rouge_p'], 4))
        headers.append('ROUGE-L R')
        values.append(round(result['rouge_r'], 4))
        headers.append('ROUGE-L F1')
        values.append(round(result['rouge_f'], 4))

    logging.info('PRINTING RESULTS...')
    print(tabulate([values], headers=headers))

    df = pd.DataFrame(data=values, index=headers)
    df['Metric'], df['Score'] = headers, values
    # df[['Metric', 'Score']].to_csv(f'./{dataset_name}/eval/csvs/{system_name}_results.csv', index=False)

    json_res = {}
    for head, val in zip(headers, values):
        json_res[head] = val
    json.dump(json_res, open(f'./{dataset_name}/eval/jsons/{system_name}_met_scores.json', 'w'), indent='\t')

    logging.info('FINISHING PRINTING RESULTS...')
    print('FINISHING PRINTING RESULTS...')
