__author__ = 'S2free'
import sys
import os
import math
# from multiprocessing import Process, Lock
# from nlp_parse import parse
#from postagger_parse import parse
from ner_relation_feature import pipeline, ffilter, transform, pipeline_test
# from pruning_heuristics import prune
# from statistic import supertype
import argparse
 
def get_number(filename):
    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
        return count
# python code/DataProcessor/feature_generation.py KBP 10 0 1.0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA', nargs='+', default=['KBP'])
    # parser.add_argument('--numOfProcesses', type = int, default = 10)
    # parser.add_argument('--negWeight', nargs='+ ', type = float, default=[1.0])
    args = parser.parse_args()

    for data in args.DATA:
        source_fold = '/shared/data/ll2/CoType/data/intermediate/'+data+'/'
        intermediate_fold = '/shared/data/ll2/CoType/data/intermediate/'+data+'/'
        train_indata = intermediate_fold + 'ntrain.json'

        test_indata = intermediate_fold + 'ntest.json'
        pipeline(train_indata, source_fold+'brown/paths', intermediate_fold)
        ffilter(intermediate_fold+'feature_map.json', intermediate_fold+'train_x.txt', intermediate_fold+'nfeature_map.json', intermediate_fold+'ntrain_x.txt', 3)

        pipeline_test(test_indata, source_fold + 'brown/paths', intermediate_fold)

        transform(intermediate_fold+'ntrain_x.txt', intermediate_fold+'train_y.txt', intermediate_fold+'train.data')
        transform(intermediate_fold+'test_x.txt', intermediate_fold+'test_y.txt', intermediate_fold+'test.data')

    # ### Perform no pruning to generate training data
    # print 'Start rm training and test data generation'
    # feature_number = get_number(outdir + '/feature.txt')
    # type_number = get_number(outdir + '/type.txt')
    # prune(outdir, outdir, 'no', feature_number, type_number, neg_label_weight=float(sys.argv[4]), isRelationMention=True, emDir=outdir_em)


    # if len(sys.argv) != 5:
    #     print 'Usage:feature_generation.py -DATA -numOfProcesses -emtypeFlag(0 or 1) -negWeight (1.0)'
    #     exit(1)
    # indir = 'data/source/%s' % sys.argv[1]
    # if int(sys.argv[3]) == 1:
    #     outdir = 'data/intermediate/%s_emtype/rm' % sys.argv[1]
    #     requireEmType = True
    # elif int(sys.argv[3]) == 0:
    #     outdir = 'data/intermediate/%s/rm' % sys.argv[1]
    #     requireEmType = False
    # else:
    #     print 'Usage:feature_generation.py -DATA -numOfProcesses -emtypeFlag(0 or 1)'
    #     exit(1)
    # outdir_em = 'data/intermediate/%s/em' % sys.argv[1]
    # # NLP parse
    # raw_train_json = indir + '/train.json'
    # raw_test_json = indir + '/test.json'
    # train_json = outdir + '/train_new.json'
    # test_json = outdir + '/test_new.json'

   
    # print 'Start rm feature extraction'
    # pipeline(train_json, indir + '/brown', outdir, requireEmType=requireEmType, isEntityMention=False)
    # filter(outdir+'/feature.map', outdir+'/train_x.txt', outdir+'/feature.txt', outdir+'/train_x_new.txt')

    # pipeline_test(test_json, indir + '/brown', outdir+'/feature.txt',outdir+'/type.txt', outdir, requireEmType=requireEmType, isEntityMention=False)

    # ### Perform no pruning to generate training data
    # print 'Start rm training and test data generation'
    # feature_number = get_number(outdir + '/feature.txt')
    # type_number = get_number(outdir + '/type.txt')
    # prune(outdir, outdir, 'no', feature_number, type_number, neg_label_weight=float(sys.argv[4]), isRelationMention=True, emDir=outdir_em)

