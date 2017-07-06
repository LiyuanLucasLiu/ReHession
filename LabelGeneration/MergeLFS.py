__author__ = 'LucasL'

import os
import math
import unicodedata
import argparse
import json
import random
import numpy as np

#re index the distant supervisions identified by -1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intput_json', nargs='+', default=['./Data/source/KBP/raw_train.json'])
    parser.add_argument('--output_json', nargs='+', default=['./Data/source/KBP/ntrain.json'])
    parser.add_argument('--output_count', nargs='+', default=['./Data/source/KBP/col_count.csv'])
    args = parser.parse_args()

    assert(len(args.intput_json)==len(args.output_json))
    assert(len(args.intput_json)==len(args.output_count))

    for find in range(0, len(args.intput_json)):
        with open(args.intput_json[find], 'r') as f:
            lines = f.readlines()
            lines = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lines))
        lines = filter(lambda t: t['relationMentions'], lines)
        cur_max_idx = max(map(lambda t: max(map(lambda x: x[3], t['relationMentions'])), lines))
        
        count = np.zeros((cur_max_idx+1, cur_max_idx+1))

        fout = open(args.output_json[find], 'w')
        for line in lines:
            nMentions = list()
            for mention in line['relationMentions']:
                mutual = True
                for ind in range(0, len(nMentions)):
                    if nMentions[ind][0][0] == mention[0][0] and nMentions[ind][0][1] == mention[0][1] and nMentions[ind][1][0] == mention[1][0] and nMentions[ind][1][1] == mention[1][1]:
                        if reduce(lambda x, y: x or y, map(lambda t: t[1] == mention[3], nMentions[ind][2])):
                            mutual = False
                            break
                        for label in nMentions[ind][2]:
                            count[label[1]][mention[3]] += 1
                            count[mention[3]][label[1]] += 1
                        count[mention[3]][mention[3]] += 1
                        nMentions[ind][2].append([mention[2], mention[3]])
                        mutual = False
                        break
                if mutual:
                    count[mention[3]][mention[3]] += 1
                    nMentions.append([mention[0], mention[1], [[mention[2], mention[3]]]])
            line['relationMentions'] = nMentions
            fout.write(json.dumps(line) + '\n')
        fout.close()

        np.savetxt(args.output_count[find], count, delimiter=',')