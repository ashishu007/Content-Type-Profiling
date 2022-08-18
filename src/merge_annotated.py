"""
use this file to merge the already labelled data with the newly labelled data
"""
import pandas as pd
import argparse, json

def convert_json_to_df(js, new_data_abs=None):
    """
    convert the json file to a pandas dataframe
    """
    df = {"Sentence": [], "B": [], "W": [], "A": [], "H": [], "Abs": []}

    for idx, item in enumerate(js):
        df['Sentence'].append(item['data']['text'])

        labels = item['annotations'][0]['result'][0]['value']['choices']

        basic_val = 1 if 'Basic' in labels else 0
        df['B'].append(basic_val)

        within_val = 1 if 'Intra-Event' in labels else 0
        df['W'].append(within_val)

        across_val = 1 if 'Inter-Event' in labels else 0
        df['A'].append(across_val)

        hard_val = 1 if 'Other' in labels else 0
        df['H'].append(hard_val)

        if new_data_abs is not None:
            df['Abs'].append(new_data_abs[idx])
        else:
            df['Abs'].append('')

    return pd.DataFrame(df)

def main(is_not_first_run=False, top_k=10, first_run_part='train', datset_name='sportsett', is_not_al=False):
    """
    main function
    """

    if is_not_al:
        print('if is_not_al')
        new_data_json = json.load(open(f'./{datset_name}/data/jsons/annotated.json', 'r'))
        new_data = convert_json_to_df(new_data_json)
        old_data = pd.read_csv(f'./{datset_name}/data/base/train.tsv', sep='\t')
        merged_data = pd.concat([old_data, new_data], ignore_index=True)
        merged_data.to_csv(f'./{datset_name}/data/tsvs/train_no_al.tsv', sep='\t', index=False)

    else:
        print('else is_not_al')

        if is_not_first_run:

            new_data_json = json.load(open(f'./{datset_name}/data/jsons/annotated.json', 'r'))
            # new_data_abs = open(f'./{datset_name}/data/txts/top_{top_k}_unlabelled_abs.txt', 'r').readlines()
            # new_data_abs = [x.strip() for x in new_data_abs]

            new_data = convert_json_to_df(new_data_json)
            # print(new_data)

            old_data = pd.read_csv(f'./{datset_name}/data/tsvs/train.tsv', sep='\t')
            merged_data = pd.concat([old_data, new_data], ignore_index=True)

            merged_data.to_csv(f'./{datset_name}/data/tsvs/train.tsv', sep='\t', index=False)

        else:
            # its the first run
            print("else")
            data_json = json.load(open(f'./{datset_name}/data/jsons/annotated.json', 'r'))
            # data_abs = open(f'./{datset_name}/data/base/starter_{first_run_part}_abs.txt', 'r').readlines()
            # data_abs = [x.strip() for x in data_abs]
            data = convert_json_to_df(data_json)#, data_abs)
            data.to_csv(f'./{datset_name}/data/base/{first_run_part}.tsv', sep='\t', index=False)
            data.to_csv(f'./{datset_name}/data/tsvs/{first_run_part}.tsv', sep='\t', index=False)

if __name__ == "__main__":
    """
    command for generating tsv from annotated data json
        python3 src/merge_annotated.py -dataset mlb -first_run_part valid/train
    
    command for merging the existing annotations (in tsv file) with the new ones (in json file)
        python3 src/merge_annotated.py -dataset mlb -not_first_run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-not_first_run", "--not_first_run", action="store_true", 
                        default=False, help="if this is not the first run")
    parser.add_argument("-not_al", "--not_al", action="store_true", 
                        default=False, help="if this is not for the AL experiment")
    parser.add_argument("-tk", "--tk", default=10, help="top k for al")
    parser.add_argument("-first_run_part", "--first_run_part", choices={'train', 'test', 'valid'}, 
                        help="train or test set for first run", default='train')
    parser.add_argument("-dataset", "--dataset", help="dataset name", default='sportsett', \
                            choices=['sportsett', 'obituary', 'sumtime', 'mlb'])

    args = parser.parse_args()
    print(args)
    main(is_not_first_run=args.not_first_run, top_k=args.tk, \
            first_run_part=args.first_run_part, datset_name=args.dataset, \
                is_not_al=args.not_al)

