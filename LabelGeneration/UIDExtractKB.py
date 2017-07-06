__author__ = 'S2free'

import os
import math
import unicodedata
import argparse
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intput_kb', nargs='+', default=['./Data/source/KBP/train.json'])
    parser.add_argument('--output_dict', nargs='+', default=['./Data/source/KBP/UIDtrain.json'])
    args = parser.parse_args()

    assert(len(args.intput_kb)==len(args.output_dict))

    for idx in range(0, len(args.intput_kb)):
        UID_look_up_table = {}
        with open(args.intput_kb[idx], 'r') as f:
            for line in f:
                ins = json.loads(line)
                if ins['articleId'] not in UID_look_up_table:
                    UID_look_up_table[ins['articleId']] = ins['relationMentions']
                else:
                    UID_look_up_table[ins['articleId']].extend(ins['relationMentions'])

        with open(args.output_dict[idx], 'w') as f:
            json.dump(UID_look_up_table, f)