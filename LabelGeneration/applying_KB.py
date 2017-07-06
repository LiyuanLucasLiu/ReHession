__author__ = 'LucasL'

import os
import math
import unicodedata
import argparse
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intput_json', nargs='+', default=['./Data/source/KBP/raw_train.json'])
    parser.add_argument('--intput_kb', nargs='+', default=['./Data/source/KBP/UIDtrain.json'])
    parser.add_argument('--output_json', nargs='+', default=['./Data/source/KBP/full_raw_train.json'])
    parser.add_argument('--save_all', action="store_true")
    parser.add_argument('--print_every', type=int, default=10000)
    args = parser.parse_args()

    assert(len(args.intput_json)==len(args.intput_kb))
    assert(len(args.intput_json)==len(args.output_json))

    for idx in range(0, len(args.intput_json)):
        with open(args.intput_json[idx], 'r') as f:
            lines = f.readlines()
            lines = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lines))
        with open(args.intput_kb[idx], 'r') as f:
            kb = json.load(f)

        print(len(lines))
        inscount = 0
        with open(args.output_json[idx], 'w') as f:
            for ins in lines:
                if inscount % args.print_every == 0:
                    print(inscount)
                inscount += 1

                relationMentions = list()
                if 'relationMentions' in ins:
                    relationMentions = ins['relationMentions']

                if ins['articleId'] in kb:
                    tokens = ins['tokens']
                    known_mentions = kb[ins['articleId']]
                    for mention in known_mentions:
                        em1 = filter(lambda t: t and not t.isspace(), mention['em1Text'].split(' '))
                        em2 = filter(lambda t: t and not t.isspace(), mention['em2Text'].split(' '))
                        em1can = list()
                        for tokenIdx in range(0, len(tokens)-len(em1)):
                            match = True
                            endIdx = tokenIdx
                            while match and endIdx < tokenIdx+len(em1):
                                if tokens[endIdx] != em1[endIdx-tokenIdx]:
                                    match = False
                                endIdx += 1
                            if match:
                                em1can.append([tokenIdx, endIdx])
                        if em1can:
                            em2can = list()
                            for tokenIdx in range(0, len(tokens)-len(em2)):
                                match = True
                                endIdx = tokenIdx
                                while match and endIdx < tokenIdx+len(em2):
                                    if tokens[endIdx] != em2[endIdx-tokenIdx]:
                                        match = False
                                    endIdx += 1
                                if match:
                                    em2can.append([tokenIdx, endIdx])
                            if em2can:
                                relationMentions.extend([[x, y, mention['label'], -1] for x in em1can for y in em2can])

                if args.save_all or relationMentions:
                    ins['relationMentions'] = relationMentions
                    f.write(json.dumps(ins)+'\n')