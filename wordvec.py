from __future__ import division
import gensim
import nltk
import random
import operator
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict

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
             ')','and','as','them','their','was','what','when','you','your',"'ll",'its','s','n','which','will','an','i','a','my','if','it','where','will','that','they','there','the','$', '000', '1', '2', '10,' 'I', 'i']

NO_STOPWORDS = []


print "Came here"
""" Get POS tag based features """
#POS = set()

#POS_DICT = {'': 0, 'PRP$': 1, 'VBG': 2, 'VBD': 3, 'VB': 26, "''": 5, 'VBP': 6, 'WDT': 7, 'JJ': 8, 'WP': 9, 'VBZ': 10, 'DT': 11, '"': 12, 'RP': 13, '$': 14, 'NN': 15, '(': 16, 'FW': 17, 'POS': 18, '.': 19, 'TO': 20, 'PRP': 21, 'RB': 22, ':': 23, 'NNS': 24, 'NNP': 25, '``': 4, 'WRB': 27, 'CC': 28, 'PDT': 30, 'RBS': 31, 'RBR': 32, 'VBN': 33, 'R': 34, 'EX': 35, 'IN': 36, 'WP$': 37, 'CD': 38, 'MD': 39, 'NNPS': 40, 'h': 41, 'NNP ': 45, 'JJS': 42, 'JJR': 43, 'SYM': 44, 's': 29, 'UH': 46, 'VBP ': 47}

#hard_rtnsl = ['through', u'shoe', 'skin', 'find', 'ground', u'discipline', u'ha', 'had', 'to', 'going', u'board', 'do', 'good', 'get', 'very', 'material', u'capsule', u'breast', 'day', 'people', u'seat', 'see', 'are', 'packed', 'out', 'even', 'for', 'crust', 'enough', 'between', 'red', 'be', 'wheat', 'dirt', 'imagine', 'carbide', 'come', 'on', 'stone', 'her', 'of', 'taking', 'keep', 'turn', 'place', 'cheese', 'into', u'one', 'down', 'fast', 'little', 'long', u'eye', 'would', 'been', 'plastic', 'much', 'way', 'taken', 'tell', u'shell', 'took', 'part', 'determination', u'line', 'believe', 'with', 'myself', 'look', 'this', 'science', 'up', 'making', u'feeling', 'study', 'is', 'surface', 'evidence', 'at', 'have', 'in', 'court', 'winter', 'no', 'make', 'reality', 'rubber', 'take', 'so', "'s", 'sided', 'enamel', 'coat', u'cover', u'face', 'edge', 'green', 'time', 'baked', 'having', u'fact', 'know']

#line_rtnls = ['taxi', 'walking', 'answering', 'deck', 'telecom', u'executive', 'hamlet', 'through', 'fishing', 'crowded', 'fine', 'profitable', 'cut', 'personal', 'lake', 'should', 'to', u'minute', u'joke', 'complaint', u'tourist', 'outside', 'food', 'jerking', 'five', 'drawn', u'walk', 'pier', 'bank', u'loss', 'like', 'cable', u'transmission', 'gender', 'motorboat', 'blurrier', 'specific', 'fisherman', 'crossing', 'hotline', 'river', 'side', 'clothes', 'draw', 'old', 'people', 'acquired', 'attached', 'fish', u'traveler', 'direct', 'blurred', u'computer', 'trading', 'are', u'sea', u'year', 'separating', 'laundry', 'racial', 'investment', 'network', 'for', 'waiting', 'profit', 'legal', 'access', 'written', 'blur', 'new', 'reading', 'across', 'blurry', 'be', u'telecommunication', 'business', 'exchange', 'sold', 'communicate', 'drew', 'water', 'busy', 'corp', 'snapped', 'along', 'by', 'tug', 'on', 'about', 'carried', 'jeep', 'of', 'industry', 'drag', 'against', 'bow', 'telesis', 'airport', 'tangled', 'stand', 'social', 'retail', 'first', 'co', 'bell', u'communication', 'into', 'private', 'one', 'hook', 'jammed', 'fast', '176', u'open', 'market', 'speak', 'standing', 'toy', 'from', 'tread', 'service', 'two', 'long', 'subscriber', 'pc', 'vax', 'call', u'vehicle', u'wait', 'checkout', u'store', 'more', 'flat', 'dialogue', 'selling', 'door', 'forming', 'company', 'formed', 'phone', 'understand', u'switchboard', 'catch', 'fastened', 'with', 'than', u'customer', 'novel', u'word', u'hour', 'these', u'car', 'non-art', u'caller', 'gasoline', 'up', u'rope', 'cast', 'crossed', 'thin', 'editorial', 'were', 'called', 'acquisition', 'toll-free', u'ad', 'toss', 'share', 'hauling', 'heard', u'say', 'pulling', 'at', 'have', 'in', 'ship', u'dealer', 'film', 'inc', 'sell', 'end', u'conversation', 'secured', 'get', 'brand', 'cross', u'actor', 'uttered', u'book', u'speech', 'catfish', 'switching', 'long-distance', u'product', 'exactly', "'s", u'price', 'ideological', u'hang', 'tied', 'tow', '000', 'pulled', 'delicate', 'such', 'blurring', 'single', 'off', 'third', 'largely', 'consumer', 'clear', u'sale', 'drawing', 'green', 'enter', 'apparel', 'buoy', 'corporate', 'divided', 'reserve']

print "Came here again"

instances1 = sense_instances(senseval.instances('hard.pos'), 'HARD1')
instances2 = sense_instances(senseval.instances('hard.pos'), 'HARD2')
instances3 = sense_instances(senseval.instances('hard.pos'), 'HARD3')

instances4 = sense_instances(senseval.instances('line.pos'), 'cord')
instances5 = sense_instances(senseval.instances('line.pos'), 'division')
instances6 = sense_instances(senseval.instances('line.pos'), 'formation')
instances7 = sense_instances(senseval.instances('line.pos'), 'product')
instances8 = sense_instances(senseval.instances('line.pos'), 'text')
instances9 = sense_instances(senseval.instances('line.pos'), 'phone')

lmtzr = WordNetLemmatizer()
print "Came here"

def modify_instance_with_CRFtag(index,filename,instances):
	j = index
	sentence =[]
	pos = -1
	tag =[]	
	for line in open(filename):
		columns = line.split(' ')
		if not line.strip():
			#print instances[j].context
			instances[j].context = zip(sentence,tag)
			#print instances[j].context
			#print '\n'
			if 'hard' in sentence:
				pos = sentence.index('hard')
			elif 'harder' in sentence:
				pos = sentence.index('harder')
			elif 'hardest' in sentence:
				pos = sentence.index('hardest')
			if pos == -1:
				print j
			instances[j].position = pos	
			sentence =[]
			pos = -1
			tag =[]	
			j=j+1
		elif len(columns) >=2:
			sentence.append(columns[0])
			tag.append(columns[1]+","+columns[2].rstrip('\n'))
	return instances

print "modifying instances"

'''
instances4 = modify_instance_with_CRFtag(102,'cord_output',instances4)
instances5 = modify_instance_with_CRFtag(102,'division_output',instances5)
instances6 = modify_instance_with_CRFtag(102,'formation_output',instances6)
instances7 = modify_instance_with_CRFtag(102,'product_output',instances7)
instances8 = modify_instance_with_CRFtag(102,'text_output',instances8)
instances9 = modify_instance_with_CRFtag(102,'phone_output',instances9)
'''

instances1 = modify_instance_with_CRFtag(202,'hard1_output',instances1)
instances2 = modify_instance_with_CRFtag(202,'hard2_output',instances2)
instances3 = modify_instance_with_CRFtag(202,'hard3_output',instances3)

print "modified instances"



def buildword2vecmodel(instances):
	sentences = []
	print "Building sentences"
	for i in range(0,len(instances)):
		sentences.append([x[0] for x in instances[i].context])

	print "building model"
	model = gensim.models.Word2Vec()
	model.build_vocab(sentences)
	model.train(sentences)
	return model





model1 = buildword2vecmodel(instances1)
print model1.similarity('hard','rock')

model2 = buildword2vecmodel(instances2)
print model2.similarity('hard','time')

model3 = buildword2vecmodel(instances3)
print model3.similarity('hard','rock')

model4 = buildword2vecmodel(instances4)
print model4.similarity('line','boat')

model5 = buildword2vecmodel(instances5)
print model5.similarity('line','draw')

model6 = buildword2vecmodel(instances6)
print model6.similarity('line','in')

model7 = buildword2vecmodel(instances7)
print model7.similarity('line','consumer')

model8 = buildword2vecmodel(instances8)
print model8.similarity('line','actor')

model9 = buildword2vecmodel(instances9)
print model9.similarity('line','telecommunications')


myfile = open("word2vec_hard_rat_CRFfeat", "w")

def writefeat2file (instances,model,myfile,sense):
	print "Writing to file"
	for i in range(0,len(instances)):
		sentence = [x[0] for x in instances[i].context]
		word_array =[0]*100
		rationale = []
		for (word,pos) in instances[i].context:
			if len(pos.split(","))==1 or pos.split(",")[1] == 'NR':
				rationale.append(word)
    		print len(sentence)
		print len(rationale)
		for word in sentence:	
			if word in model and word in rationale:
				word_array=word_array+model[word]
		myfile.write(sense)	
		for item in word_array:
  	    		myfile.write(" %s" % item)
		myfile.write("\n")
		


writefeat2file(instances1,model1,myfile,'1HARD')

writefeat2file(instances2,model2,myfile,'2HARD')

writefeat2file(instances3,model3,myfile,'3HARD')






'''

writefeat2file(instances4,model4,myfile,'1cord')

writefeat2file(instances5,model5,myfile,'2division')

writefeat2file(instances6,model6,myfile,'3formation')

writefeat2file(instances7,model7,myfile,'4product')

writefeat2file(instances8,model8,myfile,'5text')

writefeat2file(instances9,model9,myfile,'6phone')

'''

myfile.close()











