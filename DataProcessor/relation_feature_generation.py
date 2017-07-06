__author__ = 'LucasL'
import sys
import os
import math
from ner_relation_feature import pipeline, ffilter, transform, pipeline_test
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
    parser.add_argument('--Data', nargs='+', default=['KBP'])
    parser.add_argument('--Data_folder', default='./Data/intermediate/')
    parser.add_argument('--Output_folder', default='./Data/intermediate/')
    args = parser.parse_args()

    for data in args.Data:
        source_fold = args.Data_folder+data+'/'
        intermediate_fold = args.Output_folder+data+'/'
        train_indata = source_fold + 'ntrain.json'
        test_indata = source_fold + 'ntest.json'

        pipeline(train_indata, source_fold+'brown.txt', intermediate_fold)
        ffilter(intermediate_fold+'feature_map.json', intermediate_fold+'train_x.txt', intermediate_fold+'nfeature_map.json', intermediate_fold+'ntrain_x.txt', 0)

        pipeline_test(test_indata, source_fold + 'brown.txt', intermediate_fold)

        transform(intermediate_fold+'ntrain_x.txt', intermediate_fold+'train_y.txt', intermediate_fold+'train.data')
        transform(intermediate_fold+'test_x.txt', intermediate_fold+'test_y.txt', intermediate_fold+'test.data')
