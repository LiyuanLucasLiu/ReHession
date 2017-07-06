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
    parser.add_argument('--intput_json', nargs='+', default=['./Data/source/KBP/ntrain.json'])
    parser.add_argument('--output_json', nargs='+', default=['./Data/intermediate/KBP/ntrain.json'])
    args = parser.parse_args()

    assert(len(args.intput_json)==len(args.output_json))

    for (infile, outfile) in zip(args.intput_json, args.output_json):
        with open(infile, 'r') as f:
            lines = f.readlines()
            lines = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lines))
        fout = open(outfile, 'w')
        for line in lines:
            entityList = map(lambda t: t[0:2], line['entityMentions'])
            entityList.sort(key=lambda x: x[0])
            relationList = map(lambda t: [t[0][0:2], t[1][0:2], 0, t[2]], line['relationMentions'])
            for ind in range(0, len(relationList)):
                for entity in entityList:
                    if entity[0] < max(relationList[ind][0][0], relationList[ind][1][0]) and entity[0] > min(relationList[ind][0][0], relationList[ind][1][0]):
                        relationList[ind][2] += 1
            line['entityMentions'] = entityList
            line['relationMentions'] = relationList
            fout.write(json.dumps(line) + '\n')
        fout.close()
