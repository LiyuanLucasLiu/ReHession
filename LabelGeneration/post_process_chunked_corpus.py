__author__ = 'S2free'

# import sys
import os
import math
# from nlp_parse_raw import parse
# from postagger_parse import parse
# from ner_feature import pipeline, filter, pipeline_test
# from pruning_heuristics import prune
# from statistic import supertype
import unicodedata
import argparse
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', nargs='+', default=['/shared/data/ll2/CoType/data/source/KBP/chunked.religion.corpus'])
    parser.add_argument('--output_txt', nargs='+', default=['/shared/data/ll2/CoType/data/source/KBP/raw_pos_em.json'])
    args = parser.parse_args()

    assert(len(args.input_txt)==len(args.output_txt))
    for (infile, outfile) in zip(args.input_txt, args.output_txt):
        fout = open(outfile, 'w')
        with open(infile, 'r') as f:
            for line in f:
                ins = json.loads(line)
                nlist = filter(lambda t: t[0]!="None", map(lambda t: t.split(','), ins['entity_mentions']))
                if len(nlist) == 0:
                    continue
                nlist = map(lambda t: [int(t[0]), int(t[1])+int(t[0]), t[2]], nlist)
                fout.write(json.dumps({"UID": ins["UID"], "entityMentions": nlist, "pos": ins["pos"], "tokens": ins["tokens"], "articleId":ins["articleId"], "sentId":ins["sentId"]})+'\n')
        fout.close()