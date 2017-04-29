__author__ = 'wenqihe'

import re
from abstract_feature import AbstractFeature
from token_feature import HeadFeature

class PosFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        start = mention[0]['span'][1]
        end = mention[1]['span'][0]
        if mention[0]['span'][0] > mention[1]['span'][0]:
            start = mention[1]['span'][1]
            end = mention[0]['span'][0]
        for i in xrange(start, end):
            features.append('POS_%s' % sentence['pos'][i])


class DistanceFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        dist = mention[1]['span'][0] - mention[0]['span'][1]
        if mention[1]['span'][0] < mention[0]['span'][0]:
            dist = mention[0]['span'][0] - mention[1]['span'][1]
        features.append('DISTANCE_%d' % dist)

class EntityMentionOrderFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        if mention[0]['span'][0] < mention[1]['span'][0]:
            features.append('EM1_BEFORE_EM2')
        elif mention[0]['span'][0] > mention[1]['span'][0]:
            features.append('EM2_BEFORE_EM1')

class NumOfEMBetweenFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        numOfEMBetween = mention[2]
        features.append('NUM_EMS_BTWEEN_%d' % numOfEMBetween)

class SpecialPatternFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        if mention[0]['span'][1] + 1 == mention[1]['span'][0]:
            if sentence['tokens'][mention[0]['span'][1]] == 'in':
                features.append('EM1_IN_EM2')
        if mention[1]['span'][1] + 1 == mention[0]['span'][0]:
            if sentence['tokens'][mention[1]['span'][1]] == 'in':
                features.append('EM2_IN_EM1')