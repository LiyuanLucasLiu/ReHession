__author__ = 'LucasL'

import os
import math
import unicodedata
import argparse
import json
import random

#re index the distant supervisions identified by -1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intput_json', nargs='+', default=['./Data/source/KBP/full_raw_train.json'])
    parser.add_argument('--start_idx', type = int, nargs='+', default=[149])
    parser.add_argument('--output_json', nargs='+', default=['./Data/source/KBP/raw_train.json'])
    parser.add_argument('--output_new_map', nargs='+', default=['./Data/source/KBP/lf2num.json'])
    args = parser.parse_args()

    assert(len(args.start_idx)==len(args.output_json))
    assert(len(args.intput_json)==len(args.output_json))
    assert(len(args.intput_json)==len(args.output_new_map))

    for find in range(0, len(args.intput_json)):
        with open(args.intput_json[find], 'r') as f:
            lines = f.readlines()
            lines = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lines))
        # lines = filter(lambda t: t['relationMentions'], lines)
        # cur_max_idx = max(map(lambda t: max(map(lambda x: x[3], t['relationMentions'])), lines))
        # cur_max_idx += 1
        cur_max_idx = args.start_idx[find]

        new_map = {}
        fout = open(args.output_json[find], 'w')
        for line in lines:
            for idx in range(0, len(line['relationMentions'])):
                if line['relationMentions'][idx][3] < 0:
                    if line['relationMentions'][idx][3] not in new_map:
                        new_map[line['relationMentions'][idx][3]] = {}
                    if line['relationMentions'][idx][2] not in new_map[line['relationMentions'][idx][3]]:
                        new_map[line['relationMentions'][idx][3]][line['relationMentions'][idx][2]] = cur_max_idx
                        cur_max_idx += 1
                    line['relationMentions'][idx][3] = new_map[line['relationMentions'][idx][3]][line['relationMentions'][idx][2]]
            fout.write(json.dumps(line) + '\n')

        with open(args.output_new_map[find], 'w') as f:
            json.dump(new_map, f)
            