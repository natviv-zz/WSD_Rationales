WSD_Rationales
==============

Using annotator rationales for WSD - Around 1000 instances to be annotated with rationales for WSD. The words hard, line and serve from the senseval 2 corpus would be used.

The first 100 training samples for each sense of the word are annotated with rationales.
These appear as <POS_TAG,R> along with the POS TAGS comma separated

Need to have nltk installed with the senseval-2 corposes.

Within python, run:
execfile('wsd_code.py')


To get context words for a training sample 

[x[0] for x in sense_instances(senseval.instances('hard.pos'), 'HARD1')[0].context]

To get context POS tags for a training sample

[x[1] for x in sense_instances(senseval.instances('hard.pos'), 'HARD1')[0].context]

Check POS tags using split to see if tagged with raionales.
A rationale tag appears as <POS_TAG,R>

Based on this tutorial. Code cleaned up for latest vesions of NLTK - issues with FreqDist
http://www.inf.ed.ac.uk/teaching/courses/fnlp/Tutorials/7_WSD/tutorial.html

Running the Naive Bayes classifier
wst_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_word_features) 

To-do:
1> Generate rationales for line and serve
2> Add more sophisticated features
3> Code to combine features
5> Code for rationale based features
6> Integrate with scikit-learn SVM and train
7> Extending rationales to senetence boundaries
8> Code for automatic generation of rationales
