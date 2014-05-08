# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import random
import operator
import gensim
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from collections import Counter

#Code based on FNLP tutorial and modified to use with NLTK 3.0. Baseline tests code is the same.

def senses(word):
    return list(set(i.senses[0] for i in senseval.instances(word)))


    

def sense_instances(instances, sense):
    return [instance for instance in instances if instance.senses[0]==sense]


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

NEWSTOPWORDS = [',', '?', '"', '``', "''", "'", '--', '-', ';', '(',
             ')','and','as','them','their','was','what','when','you','your','its','n','which','will','an','a','my','if','it','where','will','that','they','there','the']

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


lmtzr = WordNetLemmatizer()


def ann_context_features(instance, vocab, dist=10):
    features = [0]*len(vocab)
    ind = instance.position
    con = instance.context
    rationale = []
		
    for (word,pos) in con:
	if len(pos.split(","))==1 or pos.split(",")[1] == 'NR':
		rationale.append(lmtzr.lemmatize(word))
    
					
		
    for i in range(max(0, ind-dist), ind):
	if lmtzr.lemmatize(con[i][0]) in vocab:
        	features[vocab[lmtzr.lemmatize(con[i][0])][1]] = vocab[lmtzr.lemmatize(con[i][0])][0]/(ind - i)

    for i in range(ind+1, min(ind+dist+1, len(con))):
	if lmtzr.lemmatize(con[i][0]) in vocab:
        	features[vocab[lmtzr.lemmatize(con[i][0])][1]] = vocab[lmtzr.lemmatize(con[i][0])][0]/(i - ind)

 
    #features['word'] = instance.word
    #features['pos'] = con[1][1]
    return features

def ann_pos_features(instance, pos_dict, dist=3):
    features = [0]*len(pos_dict)
    ind = instance.position
    con = instance.context

    rationale = []
    for (word,pos) in con:
	
	if len(pos.split(","))==1 or pos.split(",")[1] == 'NR':
		rationale.append(lmtzr.lemmatize(word))

    

    for i in range(max(0, ind-dist), ind):
	if con[i][1].split(',')[0] in pos_dict:
        	features[pos_dict[con[i][1].split(',')[0]]] = 1

    for i in range(ind+1, min(ind+dist+1, len(con))):
	if con[i][1].split(',')[0] in pos_dict:
        	features[pos_dict[con[i][1].split(',')[0]]] = 1

 
    #features['word'] = instance.word
    #features['pos'] = con[1][1]
    return features



def extract_vocab(instances, stopwords=STOPWORDS, n=300):
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for word in set(words) - set(stopwords):
	    word = lmtzr.lemmatize(word)	
            fd[word]+=1 
            #for sense in i.senses:
                #cfd[sense].inc(word)
    #return sorted(fd.keys()[:n+1])
    feat1 = fd.items()	
    b = {feat1[i][0]: feat1[i][1] for i in range(0, len(feat1))}
    sorted_feat = sorted(b.iteritems(), key=operator.itemgetter(1))
    return sorted_feat[-(n+1):]	

def extract_colloc_vocab(instances, stopwords=NEWSTOPWORDS, n=50):
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
	ind = i.position
	con = i.context
	dist = 5
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for j in range(max(1, ind-dist), ind):
	    if con[j][0] not in stopwords:
		word = lmtzr.lemmatize(con[j][0])		
            	fd[word]+=1	

	for j in range(ind+1, min(ind+dist+1, len(con))):
	    if con[j][0] not in stopwords:
            	word = lmtzr.lemmatize(con[j][0])		
            	fd[word]+=1
    feat1 = fd.items()
    b = {feat1[i][0]: feat1[i][1] for i in range(0, len(feat1))}
    sorted_feat = sorted(b.iteritems(), key=operator.itemgetter(1))
    return sorted_feat[-(n+1):]	
        
    
def extract_rationale_vocab(instances, stopwords=NEWSTOPWORDS, n=50):
  
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
	ind = i.position
	con = i.context
	dist = 100
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for j in range(max(1, ind-dist), ind):
	    if con[j][0] not in stopwords and len(con[j][1].split(","))>1:
		word = lmtzr.lemmatize(con[j][0])		
            	fd[word]+=1	

	for j in range(ind+1, min(ind+dist+1, len(con))):
	    if con[j][0] not in stopwords and len(con[j][1].split(","))>1:
            	word = lmtzr.lemmatize(con[j][0])		
            	fd[word]+=1
    feat1 = fd.items()
    b = {feat1[i][0]: feat1[i][1] for i in range(0, len(feat1))}
    sorted_feat = sorted(b.iteritems(), key=operator.itemgetter(1))
    return sorted_feat[-(n+1):]	
        


def normalize_dict_vocab(voc1):
	voc1 = dict(voc1)
	total = sum(voc1.values())
	for key, value in voc1.items():
    		voc1[key] = value / total
	return voc1

instances1 = sense_instances(senseval.instances('hard.pos'), 'HARD1')
sentences = []
for i in range(0,len(instances1)):
	sentences.append([x[0] for x in instances1[i].context])
model1 = gensim.models.Word2Vec(sentences)





""" DICT for context features - use stopwords """

"""
instances1 = sense_instances(senseval.instances('hard.pos'), 'HARD1')
voc1 = extract_vocab(instances1,STOPWORDS,50)
voc1 = normalize_dict_vocab(voc1)

instances2 = sense_instances(senseval.instances('hard.pos'), 'HARD2')
voc2 = extract_vocab(instances2,STOPWORDS,50)
voc2 = normalize_dict_vocab(voc2)

instances3 = sense_instances(senseval.instances('hard.pos'), 'HARD3')
voc3 = extract_vocab(instances3,STOPWORDS,50)
voc3 = normalize_dict_vocab(voc3)

hard_voc = dict(Counter(voc1)+Counter(voc2)+Counter(voc3))
maxim = max(hard_voc.values())
for key, value in hard_voc.items():
    		hard_voc[key] = value / maxim

HARDVOC_DICT = {e:(i,j) for e,i,j in zip(hard_voc.keys(),hard_voc.values(),range(len(hard_voc)))}


voc1_n = extract_colloc_vocab(instances1,NEWSTOPWORDS,40)
voc1_n = normalize_dict_vocab(voc1_n)

voc2_n = extract_colloc_vocab(instances2,NEWSTOPWORDS,40)
voc2_n = normalize_dict_vocab(voc2_n)

voc3_n = extract_colloc_vocab(instances3,NEWSTOPWORDS,40)
voc3_n = normalize_dict_vocab(voc3_n)

hard_voc_n = dict(Counter(voc1_n)+Counter(voc2_n)+Counter(voc3_n))
maxim = max(hard_voc_n.values())
for key, value in hard_voc_n.items():
    		hard_voc_n[key] = value / maxim

HARDVOC_DICT_N = {e:(i,j) for e,i,j in zip(hard_voc_n.keys(),hard_voc_n.values(),range(len(hard_voc_n)))}
"""


instances4 = sense_instances(senseval.instances('line.pos'), 'cord')
voc1 = extract_vocab(instances4,STOPWORDS,50)
voc1 = normalize_dict_vocab(voc1)

instances5 = sense_instances(senseval.instances('line.pos'), 'division')
voc2 = extract_vocab(instances5,STOPWORDS,50)
voc2 = normalize_dict_vocab(voc2)

instances6 = sense_instances(senseval.instances('line.pos'), 'formation')
voc3 = extract_vocab(instances6,STOPWORDS,50)
voc3 = normalize_dict_vocab(voc3)

instances7 = sense_instances(senseval.instances('line.pos'), 'product')
voc4 = extract_vocab(instances7,STOPWORDS,50)
voc4 = normalize_dict_vocab(voc4)

instances8 = sense_instances(senseval.instances('line.pos'), 'text')
voc5 = extract_vocab(instances8,STOPWORDS,50)
voc5 = normalize_dict_vocab(voc5)

instances9 = sense_instances(senseval.instances('line.pos'), 'phone')
voc6 = extract_vocab(instances9,STOPWORDS,50)
voc6 = normalize_dict_vocab(voc6)

line_voc = dict(Counter(voc1)+Counter(voc2)+Counter(voc3)+Counter(voc4)+Counter(voc5)+Counter(voc6))
maxim = max(line_voc.values())
for key, value in line_voc.items():
    		line_voc[key] = value / maxim

LINEVOC_DICT = {e:(i,j) for e,i,j in zip(line_voc.keys(),line_voc.values(),range(len(line_voc)))}

voc1_n = extract_colloc_vocab(instances4,NEWSTOPWORDS,40)
voc1_n = normalize_dict_vocab(voc1_n)

voc2_n = extract_colloc_vocab(instances5,NEWSTOPWORDS,40)
voc2_n = normalize_dict_vocab(voc2_n)

voc3_n = extract_colloc_vocab(instances6,NEWSTOPWORDS,40)
voc3_n = normalize_dict_vocab(voc3_n)

voc4_n = extract_colloc_vocab(instances7,NEWSTOPWORDS,40)
voc4_n = normalize_dict_vocab(voc4_n)

voc5_n = extract_colloc_vocab(instances8,NEWSTOPWORDS,40)
voc5_n = normalize_dict_vocab(voc5_n)

voc6_n = extract_colloc_vocab(instances9,NEWSTOPWORDS,40)
voc6_n = normalize_dict_vocab(voc6_n)


line_voc_n = dict(Counter(voc1_n)+Counter(voc2_n)+Counter(voc3_n)+Counter(voc4_n)+Counter(voc5_n)+Counter(voc6_n))
maxim = max(line_voc_n.values())
for key, value in line_voc_n.items():
    		line_voc_n[key] = value / maxim

LINEVOC_DICT_N = {e:(i,j) for e,i,j in zip(line_voc_n.keys(),line_voc_n.values(),range(len(line_voc_n)))}




j = 102
sentence =[]
pos = -1
tag =[]	
for line in open('cord_output'):
	columns = line.split(' ')
	if not line.strip():
		#print instances4[j].context
		instances4[j].context = zip(sentence,tag)
		#print instances4[j].context
		#print '\n'
		if 'line' in sentence:
			pos = sentence.index('line')
		elif 'lines' in sentence:
			pos = sentence.index('lines')
		if pos == -1:
			print j
		instances4[j].position = pos	
		sentence =[]
		pos = -1
		tag =[]	
		j=j+1
	elif len(columns) >=2:
		sentence.append(columns[0])
		tag.append(columns[1]+","+columns[2].rstrip('\n'))


#TODO Cleanup repeated code

"""
j = 102
sentence =[]
pos = -1
tag =[]	
for line in open('division_output'):
	columns = line.split(' ')
	if not line.strip():
		#print instances5[j].context
		instances5[j].context = zip(sentence,tag)
		#print instances5[j].context
		#print '\n'
		if 'line' in sentence:
			pos = sentence.index('line')
		elif 'lines' in sentence:
			pos = sentence.index('lines')
		if pos == -1:
			print j
		instances5[j].position = pos	
		sentence =[]
		pos = -1
		tag =[]	
		j=j+1
	elif len(columns) >=2:
		sentence.append(columns[0])
		tag.append(columns[1]+","+columns[2].rstrip('\n'))

j = 102
sentence =[]
pos = -1
tag =[]	
for line in open('formation_output'):
	columns = line.split(' ')
	if not line.strip():
		#print instances6[j].context
		instances6[j].context = zip(sentence,tag)
		#print instances6[j].context
		#print '\n'
		if 'line' in sentence:
			pos = sentence.index('line')
		elif 'lines' in sentence:
			pos = sentence.index('lines')
		if pos == -1:
			print j
		instances6[j].position = pos	
		sentence =[]
		pos = -1
		tag =[]	
		j=j+1
	elif len(columns) >=2:
		sentence.append(columns[0])
		tag.append(columns[1]+","+columns[2].rstrip('\n'))

j = 102
sentence =[]
pos = -1
tag =[]	
for line in open('product_output'):
	columns = line.split(' ')
	if not line.strip():
		#print instances7[j].context
		instances7[j].context = zip(sentence,tag)
		#print instances7[j].context
		#print '\n'
		if 'line' in sentence:
			pos = sentence.index('line')
		elif 'lines' in sentence:
			pos = sentence.index('lines')
		if pos == -1:
			print j
		instances7[j].position = pos	
		sentence =[]
		pos = -1
		tag =[]	
		j=j+1
	elif len(columns) >=2:
		sentence.append(columns[0])
		tag.append(columns[1]+","+columns[2].rstrip('\n'))

j = 102
sentence =[]
pos = -1
tag =[]	
for line in open('text_output'):
	columns = line.split(' ')
	if not line.strip():
		#print instances8[j].context
		instances8[j].context = zip(sentence,tag)
		#print instances8[j].context
		#print '\n'
		if 'line' in sentence:
			pos = sentence.index('line')
		elif 'lines' in sentence:
			pos = sentence.index('lines')
		if pos == -1:
			print j
		instances8[j].position = pos	
		sentence =[]
		pos = -1
		tag =[]	
		j=j+1
	elif len(columns) >=2:
		sentence.append(columns[0])
		tag.append(columns[1]+","+columns[2].rstrip('\n'))

j = 102
sentence =[]
pos = -1
tag =[]	
for line in open('phone_output'):
	columns = line.split(' ')
	if not line.strip():
		#print instances9[j].context
		instances9[j].context = zip(sentence,tag)
		#print instances9[j].context
		#print '\n'
		if 'line' in sentence:
			pos = sentence.index('line')
		elif 'lines' in sentence:
			pos = sentence.index('lines')
		if pos == -1:
			print j
		instances9[j].position = pos	
		sentence =[]
		pos = -1
		tag =[]	
		j=j+1
	elif len(columns) >=2:
		sentence.append(columns[0])
		tag.append(columns[1]+","+columns[2].rstrip('\n'))

"""

"""
myfile = open("output_line_CRFfeat", "w")
count = 0;
for i in instances4:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
	#rat_feat = ann_context_features(i,LINEVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	if len(feat) == 365:	
		print len(feat)
	myfile.write("1cord ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances5:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,LINEVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	if len(feat) == 365:	
		print len(feat)
	myfile.write("2division ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances6:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,LINEVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	if len(feat) == 365:	
		print len(feat)
	myfile.write("3formation ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances7:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,LINEVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	if len(feat) == 365:	
		print len(feat)
	myfile.write("4product ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances8:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,LINEVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	if len(feat) == 365:	
		print len(feat)
	myfile.write("5text ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances9:
	con_feat = ann_context_features(i,LINEVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,LINEVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,LINEVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	if len(feat) == 365:	
		print len(feat)
	myfile.write("6phone ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

print count


myfile.close()
"""
"""
myfile = open("output_hard_rat_CRFfeat", "w")
count = 0;


for i in instances1:
	con_feat = ann_context_features(i,HARDVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,HARDVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,HARDVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("1HARD ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances2:
	con_feat = ann_context_features(i,HARDVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,HARDVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,HARDVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("2HARD ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1

for i in instances3:
	con_feat = ann_context_features(i,HARDVOC_DICT,70)
	pos_feat = ann_pos_features(i,POS_DICT,5)
	word_feat = ann_context_features(i,HARDVOC_DICT_N,5)
        #rat_feat = ann_context_features(i,HARDVOC_DICT_R,70)
        feat = con_feat + pos_feat + word_feat
	print len(feat)
	myfile.write("3HARD ")	
	for item in feat:
  	    myfile.write("%s " % item)
	myfile.write("\n")
	count=count+1



myfile.close()




	
print count
"""
""" Generate CRF training and testing for rationales  """ 
"""
myfile = open("phone_train", "w")
count = 0;

for i in instances9[0:101]:
	con = i.context	
	for j in range(0,len(con)):
	    word = con[j][0]
	    if word not in NEWSTOPWORDS:	
	    	pos = con[j][1].split(',')[0]
	    	rat = 'NR' 
	    	if len(con[j][1].split(',')) ==2:
	    		rat = 'R'
 	    	myfile.write("%s %s %s" % (word, pos, rat))
	    	myfile.write("\n")
	myfile.write("\n")
	count=count+1


myfile.close()

print count

"""
"""
myfile = open("phone_test", "w")
count = 0;

for i in instances9[102:len(instances9)]:
	con = i.context	
	for j in range(0,len(con)):
	    word = con[j][0]	
	    if word not in NEWSTOPWORDS and len(con[j])>1:
	    	pos = con[j][1].split(',')[0]
	    	rat = 'NR' 
	    	if len(con[j][1].split(',')) > 1:
	    		rat = 'R'
 	    	myfile.write("%s %s" % (pos, word))
	    	myfile.write("\n")
	myfile.write("\n")
	count=count+1


myfile.close()

print count
"""


#Base line Naive Bayes Classifier test
    
def wsd_classifier(trainer, word, features, stopwords_list = STOPWORDS, number=300, log=False, distance=3, confusion_matrix=False):
    
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
        
        
    


