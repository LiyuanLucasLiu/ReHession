from Feature import *
import sys
from rel_mention_reader import MentionReader
import json
reload(sys)
sys.setdefaultencoding('utf8') 

class NERFeature(object):

    def __init__(self, is_train, brown_file, feature_mapping={}, label_mapping={}, function_mapping={}):
        self.is_train = is_train
        self.feature_list = []
        self.feature_mapping = feature_mapping # {feature_name: [feature_id, feature_frequency]}
        self.label_mapping = label_mapping 
        # {label_name: [label_id, label_func_id, label_frequency]} -- train
        # {label_name: [label_id, label_frequency]} -- test
        self.function_mapping = function_mapping

        def getCount(input_map):
            if not input_map:
                return 0
            else:
                return max(map(lambda t: t[0], input_map.values())) + 1
        self.feature_count = getCount(self.feature_mapping)
        self.label_count = getCount(self.label_mapping)
        self.function_count = getCount(self.function_mapping)

        # head feature
        self.feature_list.append(HeadFeature())
        # token feature
        self.feature_list.append(EntityMentionTokenFeature())
        self.feature_list.append(BetweenEntityMentionTokenFeature())
        # context unigram
        self.feature_list.append(ContextFeature(window_size=3))
        # context bigram
        self.feature_list.append(ContextGramFeature(window_size=3))
        # pos feature
        self.feature_list.append(PosFeature())
        # word shape feature
        self.feature_list.append(EntityMentionOrderFeature())
        # length feature
        self.feature_list.append(DistanceFeature())
        # character feature
        self.feature_list.append(NumOfEMBetweenFeature())
        self.feature_list.append(SpecialPatternFeature())
        # brown clusters
        self.feature_list.append(BrownFeature(brown_file))
            # if requireEmType:
            #     self.feature_list.append(EMTypeFeature())


    def extract(self, sentence, mention):
        # extract feature strings
        feature_str = []
        for f in self.feature_list:
            f.apply(sentence, mention, feature_str)
        #print ' '.join(sentence.tokens), feature_str
            # print f
        # map feature_names and label_names
        feature_ids = set()
        label_ids = list()
        for s in feature_str:
            if s in self.feature_mapping:
                feature_ids.add(self.feature_mapping[s][0])
                self.feature_mapping[s][1] += 1  # add frequency
            elif self.is_train:
                feature_ids.add(self.feature_count)
                self.feature_mapping[s] = [self.feature_count, 1]
                self.feature_count += 1
        #if (mention.labels) > 1:
            #print sentence.articleId, sentence.sentId
        if self.is_train:
            for l in mention.labels:
                if l[0] in self.label_mapping:
                    label_name = self.label_mapping[l[0]][0]
                    self.label_mapping[l[0]][1] += 1
                else:
                    label_name = self.label_count
                    self.label_mapping[l[0]] = [label_name, 1]
                    self.label_count += 1
                if l[1] in self.function_mapping:
                    func_name = self.function_mapping[l[1]][0]
                    self.function_mapping[l[1]][1] += 1
                else:
                    func_name = self.function_count
                    self.function_mapping[l[1]] = [func_name, 1]
                    self.function_count += 1
                label_ids.append([label_name, func_name])
        else:
            for l in mention.labels:
                if l[0] in self.label_mapping:
                    label_name = self.label_mapping[l[0]][0]
                    self.label_mapping[l[0]][1] += 1
                    label_ids.append([label_name, -1])
                else:
                    #print 'you r here'
                    print(l[0])

        return feature_ids, label_ids


def pipeline(json_file, brown_file, outdir):
    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=True, brown_file=brown_file, feature_mapping={}, label_mapping={}, function_mapping={})
    count = 0
    gx = open(outdir+'train_x.txt', 'w')
    gy = open(outdir+'train_y.txt', 'w')
    # f = open(outdir+'/feature_', 'w')
    # t = open(outdir+'/type.txt', 'w')
    # label_counts_file = open(outdir+'/label_counts.txt', 'w')

    print 'start train feature generation'
    mention_count = 0
    mentionCountByNumOfLabels = {}
    midIndex = 0
    midMap = {}
    while reader.has_next():
        if count%100 == 0:
            sys.stdout.write('process ' + str(count) + ' lines\r')
            sys.stdout.flush()
        sentence = reader.next()
        if len(sentence.pos)!=len(sentence.tokens):
            #print "error"
            sentence.pos.insert(0,'NULL')
            if len(sentence.pos)!=len(sentence.tokens):
                print "error"
                continue
        mentions = sentence.relationMentions

        for mention in mentions:
            try:
                #print sentence.UID, mention.em1Start, mention.em1End, mention.em2Start, mention.em2End
                m_id = '%s_%d_%d_%d_%d'%(sentence.UID, mention.em1Start, mention.em1End, mention.em2Start, mention.em2End)
                feature_ids, label_ids = ner_feature.extract(sentence, mention)                
                if len(label_ids) not in mentionCountByNumOfLabels:
                    mentionCountByNumOfLabels[len(label_ids)] = 1
                else:
                    mentionCountByNumOfLabels[len(label_ids)] += 1

                gx.write(str(midIndex)+'\t'+','.join([str(x) for x in 
                    feature_ids])+'\n')
                gy.write(str(midIndex)+'\t'+','.join([str(x[0])+','+str(x[1]) for x in label_ids])+'\n')
                midMap[m_id] = midIndex
                midIndex += 1
                mention_count += 1
                count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.UID, len(sentence.tokens)
                print mention
                raise
    print '\n'
    print 'mention :%d'%mention_count
    print 'feature :%d'%len(ner_feature.feature_mapping)
    print 'label :%d'%len(ner_feature.label_mapping)
    sorted_map = sorted(mentionCountByNumOfLabels.items(),cmp=lambda x,y:x[0]-y[0])
    with open(outdir + 'label_count.json', 'w') as f:
        json.dump(sorted_map, f)
    with open(outdir + 'feature_map.json', 'w') as f:
        json.dump(ner_feature.feature_mapping, f)
    with open(outdir + 'label_map.json', 'w') as f:
        json.dump(ner_feature.label_mapping, f)
    with open(outdir + 'func_map.json', 'w') as f:
        json.dump(ner_feature.function_mapping, f)
    with open(outdir + 'mid_map.json', 'w') as f:
        json.dump(midMap, f)

    reader.close()
    gx.close()
    gy.close()
    

def pipeline_test(json_file, brown_file, outdir):
    with open(outdir + 'nfeature_map.json', 'r') as f:
        featureMap = json.load(f)
    with open(outdir + 'label_map.json', 'r') as f:
        labelMap = json.load(f)
    with open(outdir + 'func_map.json', 'r') as f:
        funcMap = json.load(f)

    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=False
        , brown_file=brown_file, feature_mapping=featureMap, label_mapping=labelMap, function_mapping=funcMap)
    count = 0
    gx = open(outdir + 'test_x.txt', 'w')
    gy = open(outdir + 'test_y.txt', 'w')

    print 'start text feature generation'
    mention_count = 0
    mentionCountByNumOfLabels = {}
    midIndex = 0
    midMap = {}

    while reader.has_next():
        if count % 100 == 0:
            sys.stdout.write('process ' + str(count) + ' lines\r')
            sys.stdout.flush()
        sentence = reader.next()
        mentions = sentence.relationMentions
        for mention in mentions:
            try:
                m_id = '%s_%d_%d_%d_%d'%(sentence.UID, mention.em1Start, mention.em1End, mention.em2Start, mention.em2End)
                feature_ids, label_ids = ner_feature.extract(sentence, mention)
                if len(label_ids) not in mentionCountByNumOfLabels:
                    mentionCountByNumOfLabels[len(label_ids)] = 1
                else:
                    mentionCountByNumOfLabels[len(label_ids)] += 1
                gx.write(str(midIndex) + '\t' + ','.join([str(x) for x in feature_ids]) + '\n')
                gy.write(str(midIndex) + '\t' + ','.join([str(x[0]) + ','+str(x[1]) for x in label_ids]) + '\n' )
                midMap[m_id] = midIndex
                midIndex += 1
                mention_count += 1
                count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.UID, len(sentence.tokens)
                print mention
                raise
    print '\n'
    print 'mention :%d'%mention_count
    print 'feature :%d'%len(ner_feature.feature_mapping)
    print 'label :%d'%len(ner_feature.label_mapping)
    sorted_map = sorted(mentionCountByNumOfLabels.items(),cmp=lambda x,y:x[0]-y[0])
    with open(outdir + 'test_label_count.json', 'w') as f:
        json.dump(sorted_map, f)
    with open(outdir + 'test_mid_map.json', 'w') as f:
        json.dump(midMap, f)

    reader.close()
    gx.close()
    gy.close()


def pipeline_sample(json_file, brown_file, outdir):
    with open(outdir + 'nfeature_map.json', 'r') as f:
        featureMap = json.load(f)
    with open(outdir + 'label_map.json', 'r') as f:
        labelMap = json.load(f)
    with open(outdir + 'func_map.json', 'r') as f:
        ofuncMap = json.load(f)
        funcMap = {int(k):v for (k, v) in ofuncMap.iteritems()}

    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=True
        , brown_file=brown_file, feature_mapping=featureMap, label_mapping=labelMap, function_mapping=funcMap)
    count = 0
    gx = open(outdir + 'sample_x.txt', 'w')
    gy = open(outdir + 'sample_y.txt', 'w')

    print 'start sample feature generation'
    mention_count = 0
    mentionCountByNumOfLabels = {}
    midIndex = 0
    midMap = {}

    while reader.has_next():
        if count % 100 == 0:
            sys.stdout.write('process ' + str(count) + ' lines\r')
            sys.stdout.flush()
        sentence = reader.next()
        mentions = sentence.relationMentions
        for mention in mentions:
            try:
                m_id = '%s_%d_%d_%d_%d'%(sentence.UID, mention.em1Start, mention.em1End, mention.em2Start, mention.em2End)
                feature_ids, label_ids = ner_feature.extract(sentence, mention)
                if len(label_ids) not in mentionCountByNumOfLabels:
                    mentionCountByNumOfLabels[len(label_ids)] = 1
                else:
                    mentionCountByNumOfLabels[len(label_ids)] += 1
                gx.write(str(midIndex) + '\t' + ','.join([str(x) for x in feature_ids]) + '\n')
                gy.write(str(midIndex) + '\t' + ','.join([str(x[0]) + ','+str(x[1]) for x in label_ids]) + '\n' )
                midMap[m_id] = midIndex
                midIndex += 1
                mention_count += 1
                count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.UID, len(sentence.tokens)
                print mention
                raise
    print '\n'
    print 'mention :%d'%mention_count
    print 'feature :%d'%len(ner_feature.feature_mapping)
    print 'label :%d'%len(ner_feature.label_mapping)
    sorted_map = sorted(mentionCountByNumOfLabels.items(),cmp=lambda x,y:x[0]-y[0])
    with open(outdir + 'sample_label_count.json', 'w') as f:
        json.dump(sorted_map, f)
    with open(outdir + 'sample_mid_map.json', 'w') as f:
        json.dump(midMap, f)

    reader.close()
    gx.close()
    gy.close()
    

# def load_map(input):
#     f = open(input)
#     mapping = {}
#     for line in f:
#         seg = line.strip('\r\n').split('\t')
#         mapping[seg[0]] = [int(seg[1]), 0]
#     f.close()
#     return mapping


# def write_map(mapping, output):
#     sorted_map = sorted(mapping.items(),cmp=lambda x,y:x[1][0]-y[1][0])
#     for tup in sorted_map:
#         output.write(tup[0]+'\t'+str(tup[1][0])+'\t'+str(tup[1][1])+'\n')

def transform(input_feature, input_label, output_data):
    f = open(input_feature, 'r')
    l = open(input_label, 'r')
    o = open(output_data, 'w')
    ins_count = 0
    for feature in f:
        feature = filter(lambda t: t and not t.isspace(), feature.rstrip().split('\t'))
        label = filter(lambda t: t and not t.isspace(), l.readline().rstrip().split('\t'))
        assert(feature[0] == label[0])
        
        #print label
        if len(label)>1:
            labels = label[1].split(',')
        else:
            continue
        o.write(feature[0]+'\t')
        features = feature[1].split(',')
        o.write(str(len(features))+'\t'+str(len(labels)/2)+'\t'+feature[1]+'\t'+label[1]+'\n')
        ins_count += 1
    print(ins_count)


def ffilter(featurefile, trainfile, featureout,trainout, threshold):
    with open(featurefile, 'r') as f:
        oldFeatureMap = json.load(f)
    featuremap = {}
    old2new = {}
    count = 0

    for (k, v) in oldFeatureMap.iteritems():
        if v[1] >= threshold:
            old2new[v[0]] = count
            featuremap[k] = [count, v[1]]
            count += 1
    print 'Feature after filter: %d'%count
    with open(featureout,'w') as f:
        json.dump(featuremap, f)

    # scan the training set and filter features
    f = open(trainfile, 'r')
    g = open(trainout,'w')
    for line in f:
        seg = line.strip('\r\n').split('\t')
        # features = line.strip('\r\n').split(',')
        features = seg[1].split(',')
        newfeatures = set()
        for feature in features:
            feature = int(feature)
            if feature in old2new:
                newfeatures.add(old2new[feature])
        g.write(seg[0]+'\t'+','.join([str(x) for x in newfeatures])+'\n')
        # g.write(','.join([str(x) for x in newfeatures])+'\n')

    f.close()
    g.close()


# def write_map2(mapping, output):
#     sorted_map = sorted(mapping.items(),cmp=lambda x,y:x[1][0]-y[1][0])
#     for tup in sorted_map:
#         output.write(tup[0]+'\t'+str(tup[1][0])+'\n')

# if __name__ == "__main__":
#     if len(sys.argv) != 5:
#         print 'Usage:ner_feature.py -TRAIN_JSON -TEST_JSON -BROWN_FILE -OUTDIR'
#         exit(1)
#     train_json = sys.argv[1]
#     test_json = sys.argv[2]
#     brown_file = sys.argv[3]
#     outdir = sys.argv[4]
#     pipeline(train_json, brown_file, outdir)
#     filter(featurefile=outdir+'/feature.map', trainfile=outdir+'/train_x.txt', featureout=outdir+'/feature.txt',trainout=outdir+'/train_x_new.txt')
#     pipeline_test(test_json, brown_file, outdir+'/feature.txt',outdir+'/type.txt', outdir)
