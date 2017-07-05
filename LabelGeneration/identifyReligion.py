__author__ = 'S2free'

import os
import math
import unicodedata
import argparse
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', nargs='+', default=['/shared/data/ll2/CoType/data/source/KBP/chunked.corpus'])
    parser.add_argument('--input_dict', nargs='+', default=['/shared/data/ll2/CoType/data/source/KBP/religions.json'])
    parser.add_argument('--output_txt', nargs='+', default=['/shared/data/ll2/CoType/data/source/KBP/chunked.religion.corpus'])
    parser.add_argument('--print_every', type=int, default=20)
    args = parser.parse_args()

    assert(len(args.input_txt)==len(args.input_dict))
    assert(len(args.input_txt)==len(args.output_txt))

    for idx in range(0, len(args.input_txt)):
        with open(args.input_txt[idx], 'r') as f:
            lines = f.readlines()
            lines = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lines))
        with open(args.input_dict[idx], 'r') as f:
            wordlist = json.load(f)
        wordTree = {}
        for words in wordlist:
            words = filter(lambda t: t and not t.isspace(), words.split(' '))
            curDict = wordTree
            for word in words:
                if word not in curDict:
                    curDict[word] = {}
                curDict = curDict[word]
            curDict['<EOF>'] = len(words)
        inscount = 0
        with open(args.output_txt[idx], 'w') as f:
            for ins in lines:
                if inscount % args.print_every == 0:
                    print(inscount)
                inscount += 1
                tokens = ins['tokens']
                for ind in range(0, len(tokens)):
                    if tokens[ind] in wordTree:
                        curTree = wordTree[tokens[ind]]
                        tmpind = ind+1
                        while tmpind < len(tokens) and tokens[tmpind] in curTree:
                            if '<EOF>' in curTree:
                                ins['entity_mentions'].append(str(ind)+','+str(curTree['<EOF>'])+',RELIGION')
                            curTree = curTree[tokens[tmpind]]
                            tmpind += 1
                        if 'EOF' in curTree:
                            ins['entity_mentions'].append(str(ind)+','+str(curTree['<EOF>'])+',RELIGION')
                f.write(json.dumps(ins)+'\n')