"""

Description:
    This script aims to evaluate the output of data-to-text NLG models by 
    computing popular automatic metrics such as BLEU (two implementations), 
    METEOR, chrF++, TER and BERT-Score.
    
    ARGS:
        usage: acc_eval.py [-h] -dataset DATASET -system SYSTEM

        arguments:
          -h, --help            show this help message and exit
          -dataset DATASET, --dataset DATASET
                                dataset name 
                                (default: 'obituary')
                                (choices: {'obituary', 'mlb', 'sportsett', 'sumtime'})
          -system SYSTEM, --system SYSTEM
                                system name
                                (default: 'neural_t5')
                                (choices: {'neural_t5', 'neural_bart', 'neural_peg', 'neural_ed', 
                                            'neural_mp', 'neural_ent', 'neural_hir'})

    EXAMPLE:
        python3 eval/acc_eval.py -dataset dataset_name -system system_name

"""

import json
import argparse

def all_in_one(DATASET, SYSTEM):
    print(f'\nInside acc_eval.py file\nThis is {DATASET} dataset and {SYSTEM} system\n')
    js = json.load(open(f'{DATASET}/eval/annots/{SYSTEM}.json'))
    # js = js[:5]
    sys_acc_errs = {'name': 0, 'number': 0, 'word': 0, 'context': 0, 'not-checkable': 0, 'other': 0}
    for _, item in enumerate(js):
        acc_errs = item['annotations'][0]['result']
        for acc_err in acc_errs:
            if acc_err['value']['labels'][0] == 'NAME':
                sys_acc_errs['name'] += 1
            elif acc_err['value']['labels'][0] == 'NUMBER':
                sys_acc_errs['number'] += 1
            elif acc_err['value']['labels'][0] == 'WORD':
                sys_acc_errs['word'] += 1
            elif acc_err['value']['labels'][0] == 'CONTEXT':
                sys_acc_errs['context'] += 1
            elif acc_err['value']['labels'][0] == 'NOT CHECKABLE':
                sys_acc_errs['not-checkable'] += 1
            elif acc_err['value']['labels'][0] == 'OTHER':
                sys_acc_errs['other'] += 1
    sys_acc_errs = {k: v / len(js) for k, v in sys_acc_errs.items()}
    print(f'\n{SYSTEM} system accuracy error counts:\n{sys_acc_errs}')
    json.dump(sys_acc_errs, open(f'{DATASET}/eval/jsons/{SYSTEM}_acc_err.json', 'w'), indent='\t')

def get_err_percentage_by_err_category(js, category='Basic'):
    """
    This function returns the percentage of errors in a Type of Content by given Error Category.
    """
    acc_errs_wrong = {'name': 0, 'number': 0, 'word': 0, 'context': 0, 'not-checkable': 0, 'other': 0}
    acc_errs_right = {'name': 0, 'number': 0, 'word': 0, 'context': 0, 'not-checkable': 0, 'other': 0}
    for _, item in enumerate(js):
        acc_errs = item['annotations'][0]['result']
        for acc_err in acc_errs:
            if acc_err['value']['labels'][0] == f'Name - {category} - Right':
                acc_errs_right['name'] += 1
            elif acc_err['value']['labels'][0] == f'Name - {category} - Wrong':
                acc_errs_wrong['name'] += 1
            elif acc_err['value']['labels'][0] == f'Number - {category} - Right':
                acc_errs_right['number'] += 1
            elif acc_err['value']['labels'][0] == f'Number - {category} - Wrong':
                acc_errs_wrong['number'] += 1
            elif acc_err['value']['labels'][0] == f'Word - {category} - Right':
                acc_errs_right['word'] += 1
            elif acc_err['value']['labels'][0] == f'Word - {category} - Wrong':
                acc_errs_wrong['word'] += 1
            elif acc_err['value']['labels'][0] == f'Context - {category} - Right':
                acc_errs_right['context'] += 1
            elif acc_err['value']['labels'][0] == f'Context - {category} - Wrong':
                acc_errs_wrong['context'] += 1
            elif acc_err['value']['labels'][0] == f'NC - {category} - Right':
                acc_errs_right['not-checkable'] += 1
            elif acc_err['value']['labels'][0] == f'NC - {category} - Wrong':
                acc_errs_wrong['not-checkable'] += 1
            elif acc_err['value']['labels'][0] == f'Other - {category} - Right':
                acc_errs_right['other'] += 1
            elif acc_err['value']['labels'][0] == f'Other - {category} - Wrong':
                acc_errs_wrong['other'] += 1
    acc_err_per = {'name': 0, 'number': 0, 'word': 0, 'context': 0, 'not-checkable': 0, 'other': 0}
    for k, v in acc_errs_wrong.items():
        if acc_errs_right[k] != 0 or v != 0:
            acc_err_per[k] = float(f"{(v / (v + acc_errs_right[k]))*100:.2f}")
        else:
            acc_err_per[k] = 0
    print(f'\n{category} category wrong and right counts:\n{acc_errs_wrong} and \n{acc_errs_right} with \n{acc_err_per}%')
    return acc_err_per

def get_all_err_percentage_by_content_type(js, category='Basic'):
    """
    This function returns the percentage of errors in all Error Category by given Type of Content.
    """
    wrong, right = 0, 0
    for _, item in enumerate(js):
        acc_errs = item['annotations'][0]['result']
        for acc_err in acc_errs:
            if 'Right' in acc_err['value']['labels'][0] and category in acc_err['value']['labels'][0]:
                # print(acc_err['value']['labels'][0], "right")
                right += 1
            elif 'Wrong' in acc_err['value']['labels'][0] and category in acc_err['value']['labels'][0]:
                # print(acc_err['value']['labels'][0], "wrong")
                wrong += 1
    per = float(f"{(wrong / (wrong + right))*100:.2f}") if right != 0 or wrong != 0 else 0.0
    print(f'\n{category} category wrong and right counts:\n{wrong} and {right} with {per}%')
    return per

def get_overall_err_percentage(js):
    """
    This function returns the percentage of errors in all Error Category for all Content Type.
    """
    wrong, right = 0, 0
    for _, item in enumerate(js):
        acc_errs = item['annotations'][0]['result']
        for acc_err in acc_errs:
            if 'Right' in acc_err['value']['labels'][0]:
                right += 1
            elif 'Wrong' in acc_err['value']['labels'][0]:
                wrong += 1
    per = float(f"{(wrong / (wrong + right))*100:.2f}") if right != 0 or wrong != 0 else 0.0
    print(f'\nAll category wrong and right counts:\n{wrong} and {right} with {per}%')
    return per

def get_acc_by_cat_tok_norm(js):
    """
    This function returns the percentage of accuracy in a Type of Content by given Error Category normalised by token length.
    This will not work because - basic ones will be generated more and hence will be more mistaken. Whereas - complex ones will 
    be generated less and hence will be counted less. Even tho all the complex ones are wrong, the error score might be lower for them.
    """
    b, w, a, t, tok_len = 0, 0, 0, 0, 0
    for _, item in enumerate(js):
        tok_len += len(item['data']['ner'].split(' '))
        acc_errs = item['annotations'][0]['result']
        for acc_err in acc_errs:
            if 'Wrong' in acc_err['value']['labels'][0]:
                t += 1
                if 'Basic' in acc_err['value']['labels'][0]:
                    b += 1
                elif 'Within' in acc_err['value']['labels'][0]:
                    w += 1
                elif 'Across' in acc_err['value']['labels'][0]:
                    a += 1
    # return ((b/(tok_len))/len(js)), ((w/(tok_len))/len(js)), ((a/(tok_len))/len(js)), ((t/(tok_len))/len(js))
    return ((b/(tok_len))), ((w/(tok_len))), ((a/(tok_len))), ((t/(tok_len)))

def acc_by_cat(DATASET, SYSTEM):
    print(f"\nInside acc_eval.py file's acc_by_cat() function.\nThis is {DATASET} dataset and {SYSTEM} system\n")
    js = json.load(open(f'{DATASET}/eval/annots/{SYSTEM}.json'))
    # js = js[:5]
    basic = get_all_err_percentage_by_content_type(js, 'Basic')
    within = get_all_err_percentage_by_content_type(js, 'Within')
    across = get_all_err_percentage_by_content_type(js, 'Across')
    all = get_overall_err_percentage(js)
    # bt, wt, at, tt = get_acc_by_cat_tok_norm(js)
    errs = {'basic': basic, 'within': within, 'across': across, 'overall': all} #, \
            # 'basic_norm': bt, 'within_norm': wt, 'across_norm': at, 'overall_norm': tt}
    print(f'\n{SYSTEM} system accuracy error counts:\n{errs}')
    json.dump(errs, open(f'{DATASET}/eval/jsons/{SYSTEM}_acc_by_cat.json', 'w'), indent='\t')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--dataset', '-dataset', type=str, default='obituary')
    argParser.add_argument('--system', '-system', type=str, default='neural_bart')

    args = argParser.parse_args()
    print(args)

    acc_by_cat(args.dataset, args.system)
    print('\nDone!\n')
