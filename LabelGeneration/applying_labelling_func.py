__author__ = 'LucasL'

import os
import math
import unicodedata
import argparse
import json
import random

def match_tokens(tokens, poses, entities, pT, relationMentions):
    for idx1 in range(1, len(tokens) - 1):
        if tokens[idx1] in pT:
            find_end(tokens, poses, entities, idx1, idx1+1, pT[tokens[idx1]], relationMentions)

def find_end(tokens, poses, entities, first_idx, cur_idx, cur_pT, relationMentions):
    if '[EOF]' in cur_pT:
        patterns = cur_pT['[EOF]']
        for pattern in patterns:
            if pattern['reserved'] == '1':
                first_type = pattern['Type2']
                second_type = pattern['Type1']
            else:
                first_type = pattern['Type1']
                second_type = pattern['Type2']
            first_entities = filter(lambda t: t[1] <= first_idx and t[2] == first_type, entities)
            if first_entities:
                second_entities = filter(lambda t: t[0] >= cur_idx and t[2] == second_type, entities)
                if second_entities:
                    leng = pattern['rule'][0]
                    if leng != 'n' and int(leng) < len(first_entities):
                        first_entities.sort(key=lambda x: x[1])
                        first_entities = first_entities[-int(leng):]
                    leng = pattern['rule'][1]
                    if leng != 'n' and int(leng) < len(second_entities):
                        second_entities.sort(key=lambda x: x[0])
                        second_entities = second_entities[:int(leng)]
                    if pattern['reserved'] == '1':
                        first_entities, second_entities = second_entities, first_entities
                    relationMentions.extend([[x, y, pattern['relationType'], pattern['PID']] for x in first_entities for y in second_entities])
    if cur_idx < len(tokens) - 1:
        if tokens[cur_idx] in cur_pT:
            find_end(tokens, poses, entities, first_idx, cur_idx+1, cur_pT[tokens[cur_idx]], relationMentions)

        keys = filter(lambda t: t and t[0] == '<', cur_pT.keys())
        if keys:
            for key in keys:
                find_end(tokens, poses, entities, first_idx, cur_idx, cur_pT[key], relationMentions)
            if len(poses[cur_idx]) > 1:
                pos_key = '<'+poses[cur_idx][0:2]+'>'
                if pos_key in keys:
                    find_end(tokens, poses, entities, first_idx, cur_idx+1, cur_pT, relationMentions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--intput_json', nargs='+', default=['./Data/source/KBP/raw_pos_em.json'])
    parser.add_argument('--intput_lfs', nargs='+', default=['./Data/source/KBP/nlf.json'])
    parser.add_argument('--output_json', nargs='+', default=['./Data/source/KBP/raw_train.json'])
    parser.add_argument('--save_all', action="store_true")
    parser.add_argument('--print_every', type=int, default=10000)
    args = parser.parse_args()

    assert(len(args.intput_json)==len(args.intput_lfs))
    assert(len(args.intput_json)==len(args.output_json))

    for idx in range(0, len(args.intput_json)):
        with open(args.intput_json[idx], 'r') as f:
            lines = f.readlines()
            lines = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lines))
        with open(args.intput_lfs[idx], 'r') as f:
            lfs = f.readlines()
            lfs = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lfs))

        patternTree = {}
        for pattern in lfs:
            words = filter(lambda t: t and not t.isspace(), pattern['Texture'].split(' '))
            curDict = patternTree
            for word in words:
                if word not in curDict:
                    curDict[word] = {}
                curDict = curDict[word]
            if '[EOF]' not in curDict:
                curDict['[EOF]'] = [pattern]
            else:
                curDict['[EOF]'].append(pattern)

        inscount = 0
        with open(args.output_json[idx], 'w') as f:
            for ins in lines:
                if inscount % args.print_every == 0:
                    print(inscount)
                inscount += 1

                entityMentions = ins['entityMentions']
                poses = ins['pos']
                tokens = ins['tokens']
                relationMentions = list()
                match_tokens(tokens, poses, entityMentions, patternTree, relationMentions)
                if args.save_all or relationMentions:
                    ins['relationMentions'] = relationMentions
                    f.write(json.dumps(ins)+'\n')