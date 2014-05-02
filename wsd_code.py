# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import random
import operator
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from collections import defaultdict

# The following shows how the senseval corpus consists of instances, where each instance
# consists of a target word (and its tag), it position in the sentence it appeared in
# within the corpus (that position being word position, minus punctuation), and the context,
# which is the words in the sentence plus their tags.
#
# senseval.instances()[:1]
# [SensevalInstance(word='hard-a', position=20, context=[('``', '``'), ('he', 'PRP'),
# ('may', 'MD'), ('lose', 'VB'), ('all', 'DT'), ('popular', 'JJ'), ('support', 'NN'),
#Â (',', ','), ('but', 'CC'), ('someone', 'NN'), ('has', 'VBZ'), ('to', 'TO'),
# ('kill', 'VB'), ('him', 'PRP'), ('to', 'TO'), ('defeat', 'VB'), ('him', 'PRP'),
# ('and', 'CC'), ('that', 'DT'), ("'s", 'VBZ'), ('hard', 'JJ'), ('to', 'TO'), ('do', 'VB'),
# ('.', '.'), ("''", "''")], senses=('HARD1',))]

def senses(word):
    """
    This takes a target word from senseval-2 (find out what the possible
    are by running senseval.fileides()), and it returns the list of possible 
    senses for the word
    """
    return list(set(i.senses[0] for i in senseval.instances(word)))


    

def sense_instances(instances, sense):
    """
    This returns the list of instances in instances that have the sense sense
    """
    return [instance for instance in instances if instance.senses[0]==sense]

# >>> sense3 = sense_instances(senseval.instances('hard.pos'), 'HARD3')
# >>> sense3[:2]
# [SensevalInstance(word='hard-a', position=15,
#  context=[('my', 'PRP$'), ('companion', 'NN'), ('enjoyed', 'VBD'), ('a', 'DT'), ('healthy', 'JJ'), ('slice', 'NN'), ('of', 'IN'), ('the', 'DT'), ('chocolate', 'NN'), ('mousse', 'NN'), ('cake', 'NN'), (',', ','), ('made', 'VBN'), ('with', 'IN'), ('a', 'DT'), ('hard', 'JJ'), ('chocolate', 'NN'), ('crust', 'NN'), (',', ','), ('topping', 'VBG'), ('a', 'DT'), ('sponge', 'NN'), ('cake', 'NN'), ('with', 'IN'), ('either', 'DT'), ('strawberry', 'NN'), ('or', 'CC'), ('raspberry', 'JJ'), ('on', 'IN'), ('the', 'DT'), ('bottom', 'NN'), ('.', '.')],
#  senses=('HARD3',)),
#  SensevalInstance(word='hard-a', position=5,
#  context=[('``', '``'), ('i', 'PRP'), ('feel', 'VBP'), ('that', 'IN'), ('the', 'DT'), ('hard', 'JJ'), ('court', 'NN'), ('is', 'VBZ'), ('my', 'PRP$'), ('best', 'JJS'), ('surface', 'NN'), ('overall', 'JJ'), (',', ','), ('"', '"'), ('courier', 'NNP'), ('said', 'VBD'), ('.', '.')],
# senses=('HARD3',))]


_inst_cache = {}

STOPWORDS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$', '000', '1', '2', '10,' 'I', 'i', 'a', 'about', 'after', 'all', 'also', 'an', 'any',
             'are', 'as', 'at', 'and', 'be', 'being', 'because', 'been', 'but', 'by',
             'can', "'d", 'did', 'do', "don'", 'don', 'for', 'from', 'had','has', 'have', 'he',
             'her','him', 'his', 'how', 'if', 'is', 'in', 'it', 'its', "'ll", "'m", 'me',
             'more', 'my', 'n', 'no', 'not', 'of', 'on', 'one', 'or', "'re", "'s", "s",
             'said', 'say', 'says', 'she', 'so', 'some', 'such', "'t", 'than', 'that', 'the',
             'them', 'they', 'their', 'there', 'this', 'to', 'up', 'us', "'ve", 'was', 'we', 'were',
             'what', 'when', 'where', 'which', 'who', 'will', 'with', 'years', 'you',
             'your']

NEWSTOPWORDS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')','and','as','them','their','was','what','when','you','your',"'ll",'its','s','n','which','will','an','i','a','my','if','it','where','will','that','they','there','the']

NO_STOPWORDS = []



""" Get POS tag based features """
POS = set()
"""
for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('hard.pos'), 'HARD1')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('hard.pos'), 'HARD2')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('hard.pos'), 'HARD3')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('line.pos'), 'cord')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('line.pos'), 'text')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('line.pos'), 'divison')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('line.pos'), 'formation')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('line.pos'), 'phone')[i].context])

for i in range(0,101):
	POS.update([x[1].split(',')[0] for x in sense_instances(senseval.instances('line.pos'), 'product')[i].context])


POS_DICT = {e:i for e,i in zip(POS,range(0,len(POS)))}
"""
POS_DICT = {'': 0, 'PRP$': 1, 'VBG': 2, 'VBD': 3, 'VB': 26, "''": 5, 'VBP': 6, 'WDT': 7, 'JJ': 8, 'WP': 9, 'VBZ': 10, 'DT': 11, '"': 12, 'RP': 13, '$': 14, 'NN': 15, '(': 16, 'FW': 17, 'POS': 18, '.': 19, 'TO': 20, 'PRP': 21, 'RB': 22, ':': 23, 'NNS': 24, 'NNP': 25, '``': 4, 'WRB': 27, 'CC': 28, 'PDT': 30, 'RBS': 31, 'RBR': 32, 'VBN': 33, 'R': 34, 'EX': 35, 'IN': 36, 'WP$': 37, 'CD': 38, 'MD': 39, 'NNPS': 40, 'h': 41, 'NNP ': 45, 'JJS': 42, 'JJR': 43, 'SYM': 44, 's': 29, 'UH': 46, 'VBP ': 47}


def wsd_context_features(instance, vocab, dist=10):
    features = {}
    ind = instance.position
    con = instance.context
    for i in range(max(0, ind-dist), ind):
        j = ind-i
        features['left-context-word-%s(%s)' % (j, con[i][0])] = True

    for i in range(ind+1, min(ind+dist+1, len(con))):
        j = i-ind
        features['right-context-word-%s(%s)' % (j, con[i][0])] = True

 
    features['word'] = instance.word
    features['pos'] = con[1][1]
    return features

def ann_context_features(instance, vocab, dist=10):
    features = [0]*len(vocab)
    ind = instance.position
    con = instance.context
    rationale = []	
    for (word,pos) in con:
	if len(pos.split(","))==1:
		rationale.append(word)

	
    for i in range(max(0, ind-dist), ind):
	if con[i][0] in vocab and con[i][0] in rationale:
        	features[vocab[con[i][0]]] = 1

    for i in range(ind+1, min(ind+dist+1, len(con))):
	if con[i][0] in vocab and con[i][0] in rationale:
        	features[vocab[con[i][0]]] = 1

 
    #features['word'] = instance.word
    #features['pos'] = con[1][1]
    return features

def ann_pos_features(instance, pos_dict, dist=3):
    features = [0]*len(pos_dict)
    ind = instance.position
    con = instance.context
    rationale = []
    for (word,pos) in con:
	if len(pos.split(","))==1:
		rationale.append(word)

    

    for i in range(max(0, ind-dist), ind):
	if con[i][1].split(',')[0] in pos_dict and con[i][0] in rationale:
        	features[pos_dict[con[i][1].split(',')[0]]] = 1

    for i in range(ind+1, min(ind+dist+1, len(con))):
	if con[i][1].split(',')[0] in pos_dict and con[i][0] in rationale:
        	features[pos_dict[con[i][1].split(',')[0]]] = 1

 
    #features['word'] = instance.word
    #features['pos'] = con[1][1]
    return features



def wsd_word_features(instance, vocab, dist=3):
    """
    Create a featureset where every key returns False unless it occurs in the
    instance's context
    """
    features = defaultdict(lambda:False)
    features['alwayson'] = True
    #cur_words = [w for (w, pos) in i.context]
    try:
        for(w, pos) in instance.context:
            if w in vocab:
                features[w] = True
    except ValueError:
        pass
    return features


def extract_vocab(instances, stopwords=STOPWORDS, n=300):
    """
    Given a list of senseval instances, return a list of the n most frequent words that
    appear in the context of instances.  The context is the sentence that the target word
    appeared in within the corpus.
    """
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for word in set(words) - set(stopwords):
            fd[word]+=1 
            #for sense in i.senses:
                #cfd[sense].inc(word)
    #return sorted(fd.keys()[:n+1])
    feat1 = fd.items()
    b = {feat1[i][0]: feat1[i][1] for i in range(0, len(feat1))}
    sorted_feat = sorted(b.iteritems(), key=operator.itemgetter(1))
    """print sorted_feat[n+1:]"""	
    return sorted_feat[-(n+1):]	

def extract_colloc_vocab(instances, stopwords=NEWSTOPWORDS, n=50):
    """
    Given a list of senseval instances, return a list of the n most frequent words that
    appears in its context (i.e., the sentence with the target word in), output is in order
    of frequency and includes also the number of instances in which that key appears in the
    context of instances.
    """
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
	ind = i.position
	con = i.context
	dist = 3
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for j in range(max(1, ind-dist), ind):
	    if con[j][0] not in stopwords:		
            	fd[con[j][0]]+=1	

	for j in range(ind+1, min(ind+dist+1, len(con))):
	    if con[j][0] not in stopwords:
            	fd[con[j][0]]+=1
    feat1 = fd.items()
    b = {feat1[i][0]: feat1[i][1] for i in range(0, len(feat1))}
    sorted_feat = sorted(b.iteritems(), key=operator.itemgetter(1))
    return sorted_feat[-(n+1):]	
        
    










""" DICT for context features - use stopwords """


instances1 = sense_instances(senseval.instances('hard.pos'), 'HARD1')
voc1 = extract_vocab(instances1[0:201],STOPWORDS,50)
voc1 = [voc1[i][0] for i in range(0, 50)]

voc1_n = extract_colloc_vocab(instances1[0:201],NEWSTOPWORDS,40)
voc1_n = [voc1_n[i][0] for i in range(0, 40)]


instances2 = sense_instances(senseval.instances('hard.pos'), 'HARD2')
voc2 = extract_vocab(instances2[0:201],STOPWORDS,50)
voc2 = [voc2[i][0] for i in range(0, 50)]

voc2_n = extract_colloc_vocab(instances2[0:201],NEWSTOPWORDS,40)
voc2_n = [voc2_n[i][0] for i in range(0, 40)]


instances3 = sense_instances(senseval.instances('hard.pos'), 'HARD3')
voc3 = extract_vocab(instances3[0:201],STOPWORDS,50)
voc3 = [voc3[i][0] for i in range(0, 50)]

voc3_n = extract_colloc_vocab(instances3[0:201],NEWSTOPWORDS,40)
voc3_n = [voc3_n[i][0] for i in range(0, 40)]

hard_voc = set(voc1+voc2+voc3)
HARDVOC_DICT = {e:i for e,i in zip(hard_voc,range(len(hard_voc)))}

hard_voc_n = set(voc1_n+voc2_n+voc3_n)
HARDVOC_DICT_N = {e:i for e,i in zip(hard_voc_n,range(len(hard_voc_n)))}



instances4 = sense_instances(senseval.instances('line.pos'), 'cord')
voc4 = extract_vocab(instances4[0:101],STOPWORDS,50)
voc4 = [voc4[i][0] for i in range(0, 50)]

voc4_n = extract_colloc_vocab(instances4[0:101],NEWSTOPWORDS,40)
voc4_n = [voc4_n[i][0] for i in range(0, 40)]




instances5 = sense_instances(senseval.instances('line.pos'), 'division')
voc5 = extract_vocab(instances5[0:101],STOPWORDS,50)
voc5 = [voc5[i][0] for i in range(0, 50)]

voc5_n = extract_colloc_vocab(instances5[0:101],NEWSTOPWORDS,40)
voc5_n = [voc5_n[i][0] for i in range(0, 40)]


instances6 = sense_instances(senseval.instances('line.pos'), 'formation')
voc6 = extract_vocab(instances6[0:101],STOPWORDS,50)
voc6 = [voc6[i][0] for i in range(0, 50)]

voc6_n = extract_colloc_vocab(instances6[0:101],NEWSTOPWORDS,40)
voc6_n = [voc6_n[i][0] for i in range(0, 40)]



instances7 = sense_instances(senseval.instances('line.pos'), 'product')
voc7 = extract_vocab(instances7[0:101],STOPWORDS,50)
voc7 = [voc7[i][0] for i in range(0, 50)]

voc7_n = extract_colloc_vocab(instances7[0:101],NEWSTOPWORDS,40)
voc7_n = [voc7_n[i][0] for i in range(0, 40)]


instances8 = sense_instances(senseval.instances('line.pos'), 'text')
voc8 = extract_vocab(instances8[0:101],STOPWORDS,50)
voc8 = [voc8[i][0] for i in range(0, 50)]

voc8_n = extract_colloc_vocab(instances8[0:101],NEWSTOPWORDS,40)
voc8_n = [voc8_n[i][0] for i in range(0, 40)]


instances9 = sense_instances(senseval.instances('line.pos'), 'phone')
voc9 = extract_vocab(instances9[0:101],STOPWORDS,50)
voc9 = [voc9[i][0] for i in range(0, 50)]


voc9_n = extract_colloc_vocab(instances9[0:101],NEWSTOPWORDS,40)
voc9_n = [voc9_n[i][0] for i in range(0, 40)]


line_voc = set(voc4+voc5+voc6+voc7+voc8+voc9)
LINEVOC_DICT = {e:i for e,i in zip(line_voc,range(len(line_voc)))}

line_voc_n = set(voc4_n+voc5_n+voc6_n+voc7_n+voc8_n+voc9_n)
LINEVOC_DICT_N = {e:i for e,i in zip(line_voc_n,range(len(line_voc_n)))}


myfile = open("output_line_feat_rtnls", "w")
count = 0;
for i in instances4[0:101]:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("1cord ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances5[0:101]:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("2division ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances6[0:101]:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("3formation ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances7[0:101]:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("4product ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances8[0:101]:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("5text ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances9[0:101]:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("6phone ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

print count


myfile.close()

myfile = open("output_hard_feat_rtnls", "w")
count = 0;


for i in instances1[0:201]:
	con_feat = ann_context_features(i,HARDVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,HARDVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("1HARD ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances2[0:201]:
	con_feat = ann_context_features(i,HARDVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,HARDVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("2HARD ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances3[0:201]:
	con_feat = ann_context_features(i,HARDVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,HARDVOC_DICT_N,5)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("3HARD ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1









	
print count
""" DICT for word features - dont use stopwords """ 



##def wst_classifier(trainer, word, features,number=300):
##    print "Reading data..."
##    global _inst_cache
##    if word not in _inst_cache:
##        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
##    events = _inst_cache[word][:]
##    senses = list(set(l for (i, l) in events))
##    instances = [i for (i, l) in events]
##    vocab = extract_vocab(instances, n=number)
##    print ' Senses: ' + ' '.join(senses)
##
##    # Split the instances into a training and test set,
##    #if n > len(events): n = len(events)
##    n = len(events)
##    random.seed(5444522)
##    random.shuffle(events)
##    training_data = events[:int(0.8 * n)]
##    test_data = events[int(0.8 * n):n]
##    # Train classifier
##    print 'Training classifier...'
##    classifier = trainer([(features(i, vocab), label) for (i, label) in training_data])
##    # Test classifier
##    print 'Testing classifier...'
##    acc = accuracy(classifier, [(features(i, vocab), label) for (i, label) in test_data] )
##    print 'Accuracy: %6.4f' % acc

    
def wst_classifier(trainer, word, features, stopwords_list = STOPWORDS, number=300, log=False, distance=3, confusion_matrix=False):
    """
    This function takes as arguments:
        a trainer (e.g., NaiveBayesClassifier.train);
        a target word from senseval2 (you can find these out with senseval.fileids(),
            and they are 'hard.pos', 'interest.pos', 'line.pos' and 'serve.pos');
        a feature set (this can be wsd_context_features or wsd_word_features);
        a number (defaults to 300), which determines for wsd_word_features the number of
            most frequent words within the context of a given sense that you use to classify examples;
        a distance (defaults to 3) which determines the size of the window for wsd_context_features (if distance=3, then
            wsd_context_features gives 3 words and tags to the left and 3 words and tags to
            the right of the target word);
        log (defaults to false), which if set to True outputs the errors into a file errors.txt
        confusion_matrix (defaults to False), which if set to True prints a confusion matrix.

    Calling this function splits the senseval data for the word into a training set and a test set (the way it does
    this is the same for each call of this function, because the argument to random.seed is specified,
    but removing this argument would make the training and testing sets different each time you build a classifier).

    It then trains the trainer on the training set to create a classifier that performs WSD on the word,
    using features (with number or distance where relevant).

    It then tests the classifier on the test set, and prints its accuracy on that set.

    If log==True, then the errors of the classifier over the test set are written to errors.txt.
    For each error four things are recorded: (i) the example number within the test data (this is simply the index of the
    example within the list test_data); (ii) the sentence that the target word appeared in, (iii) the
    (incorrect) derived label, and (iv) the gold label.

    If confusion_matrix==True, then calling this function prints out a confusion matrix, where each cell [i,j]
    indicates how often label j was predicted when the correct label was i (so the diagonal entries indicate labels
    that were correctly predicted).
    """
    print "Reading data..."
    global _inst_cache
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    print ' Senses: ' + ' '.join(senses)

    # Split the instances into a training and test set,
    #if n > len(events): n = len(events)
    n = len(events)
    random.seed(5444522)
    random.shuffle(events)
    training_data = events[:int(0.8 * n)]
    test_data = events[int(0.8 * n):n]
    # Train classifier
    print 'Training classifier...'
    classifier = trainer([(features(i, vocab, distance), label) for (i, label) in training_data])
    # Test classifier
    print 'Testing classifier...'
    acc = accuracy(classifier, [(features(i, vocab, distance), label) for (i, label) in test_data] )
    print 'Accuracy: %6.4f' % acc
    if log==True:
        #write error file
        print 'Writing errors to errors.txt'
        output_error_file = open('errors.txt', 'w')
        errors = []
        for (i, label) in test_data:
            guess = classifier.classify(features(i, vocab, distance))
            if guess != label:
                con =  i.context
                position = i.position
                item_number = str(test_data.index((i, label)))
                word_list = []
                for (word, tag) in con:
                    word_list.append(word)
                hard_highlighted = word_list[position].upper()
                word_list_highlighted = word_list[0:position] + [hard_highlighted] + word_list[position+1:]
                sentence = ' '.join(word_list_highlighted)
                errors.append([item_number, sentence, guess,label])
        error_number = len(errors)
        output_error_file.write('There are ' + str(error_number) + ' errors!' + '\n' + '----------------------------' +
                                '\n' + '\n')
        for error in errors:
            output_error_file.write(str(errors.index(error)+1) +') ' + 'example number: ' + error[0] + '\n' +
                                    '    sentence: ' + error[1] + '\n' +
                                    '    guess: ' + error[2] + ';  label: ' + error[3] + '\n' + '\n')
        output_error_file.close()
    if confusion_matrix==True:
        gold = [label for (i, label) in test_data]
        derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        cm = nltk.ConfusionMatrix(gold,derived)
        print cm
        return cm
        
        
    
def demo():
    print "NB, with features based on 300 most frequent context words"
    wst_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_word_features)
    print
    print "NB, with features based word + pos in 6 word window"
    wst_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_context_features)
    print
##    print "MaxEnt, with features based word + pos in 6 word window"
##    wst_classifier(MaxentClassifier.train, 'hard.pos', wsd_context_features)
    


# Frequency Baseline
##hard_sense_fd = nltk.FreqDist([i.senses[0] for i in senseval.instances('hard.pos')])
##most_frequent_hard_sense= hard_sense_fd.keys()[0]
##frequency_hard_sense_baseline = hard_sense_fd.freq(hard_sense_fd.keys()[0])

##>>> frequency_hard_sense_baseline
##0.79736902838679902

