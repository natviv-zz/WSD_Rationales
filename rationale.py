# Aron Yu
# Perform Contrastive Learning using SVM
# - optimize parameters on held out data

import time
import numpy as np
from sklearn import svm
from sklearn import cross_validation

#======================#
#== EXTRACT FEATURES ==#
#======================#
print('Extracting features & labels...'),
homedir = 'C:\\Users\\Aron\\Dropbox\\Graduate School\\CS 388\\Project\\RationaleWSD\\'
#homedir = 'E:\\Dropbox\\Graduate School\\CS 388\\Project\\RationaleWSD\\'
featsdir = 'features\\'
resultdir = 'results\\'

# Inputs
word = 'hard'
targetSense = 3
numSplits = 5
postfixReg = '_reg_CRFfeat'
postfixRat = '_rat_CRFfeat'
savefile = 1

if word == 'hard':
    dataSize = 4333 #603
    featLen = 99
else:
    dataSize = 4146 #606
    featLen = 99

fileReg = 'word2vec_' + word + postfixReg
fileRat = 'word2vec_' + word + postfixRat

# Form Feature & Label Vectors for Regular Data
labelsReg = np.zeros((dataSize,))
featsReg = np.zeros((dataSize,featLen))

with open(homedir+featsdir+fileReg, 'r') as tf:
    cnt = 0
    for line in tf:
        content = line.split(' ')
        labelsReg[cnt] = content[0][0]
        featsReg[cnt,:] = content[1:len(content)-1]
        cnt += 1

# Form Feature & Label Vectors for Rationale Data
labelsRat = np.zeros((dataSize,))
featsRat = np.zeros((dataSize,featLen))

with open(homedir+featsdir+fileRat, 'r') as tf:
    cnt = 0
    for line in tf:
        content = line.split(' ')
        labelsRat[cnt] = content[0][0]
        featsRat[cnt,:] = content[1:len(content)-1]
        cnt += 1

print 'Done!'
print 'Word: ' + word + '(' + str(targetSense) + ')' + '\n'

# Form Rationale Feature Vectors
featsRat = featsReg - featsRat

# Sense Selection
labelsReg[labelsReg!=targetSense] = 0
labelsReg[labelsReg==targetSense] = 1
labelsRat[labelsRat!=targetSense] = 0
labelsRat[labelsRat==targetSense] = 1


#=================#
#== EXPERIMENTS ==#
#=================#

numHold = 100
numTest = 100

# Possible Parameters
#C_list = [1, 10, 30, 50, 80, 100]
#u_list = [0.1, 0.5, 1, 1.5]
#Ccon_list = [0.05, 0.1, 0.3, 0.5, 0.8]
C_list = [1, 10, 50, 100, 500]
u_list = [0.1, 0.5, 1, 1.5]
Ccon_list = [0.05, 0.1, 0.3, 0.5, 0.8]

# Repeat Full Experiment X Times

ratAcc = np.zeros(numSplits);
regAcc = np.zeros(numSplits);

for splitID in range(numSplits):
    
    startIter = time.clock()
    
    # Form Hold-out Set (same for each iteration)
    mainFeats, holdFeats, mainLabels, holdLabels = cross_validation.train_test_split(
                                                        featsReg, labelsReg, test_size=numHold, random_state=100)
    mainFeatsRat, holdFeatsRat, mainLabelsRat, holdLabelsRat = cross_validation.train_test_split(
                                                        featsRat, labelsRat, test_size=numHold, random_state=100)
    
    # Form Train-Test Sets
    trainFeats, testFeats, trainLabels, testLabels = cross_validation.train_test_split(
                                                        mainFeats, mainLabels, test_size=numTest, random_state=splitID)
    trainFeatsRat, testFeatsRat, trainLabelsRat, testLabelsRat = cross_validation.train_test_split(
                                                        mainFeatsRat, mainLabelsRat, test_size=numTest, random_state=splitID)

    # Modify Feature Vectors (prefix by 1 or 0)
    trainFeatsMod = np.concatenate([np.ones((trainFeats.shape[0],1)), trainFeats],1)
    testFeatsMod = np.concatenate([np.ones((testFeats.shape[0],1)), testFeats],1)
    holdFeatsMod = np.concatenate([np.ones((holdFeats.shape[0],1)), holdFeats],1)
    trainFeatsRatMod = np.concatenate([np.zeros((trainFeatsRat.shape[0],1)), trainFeatsRat],1)
    
    startCV = time.clock()
     
    # Iterate Through All Parameters
    holdAcc = np.zeros((len(C_list), len(u_list), len(Ccon_list)))
    for c in range(len(Ccon_list)):
        for x in range(len(C_list)):
            for y in range(len(u_list)):
                     
                # Concatenate Regular & Rationale Data for Training
                ctTrainFeats = np.concatenate([trainFeatsMod, trainFeatsRatMod/u_list[y]])
                ctTrainLabels = np.concatenate([trainLabels, trainLabelsRat])
                W = np.concatenate([np.ones(trainLabels.shape), Ccon_list[c]*np.ones(trainLabelsRat.shape)])
                     
                # Contrastive SVM
                modelHold = svm.SVC(C=C_list[x], kernel='linear', cache_size=2000, verbose=False)
                modelHold.fit(ctTrainFeats, ctTrainLabels, W)
                resultHold = modelHold.predict(holdFeatsMod)
                holdAcc[x,y,c] = sum(resultHold==holdLabels) / float(len(holdLabels))
              
            print 'Done with C = ' + str(C_list[x])
       
    elapsedCV = (time.clock() - startCV)
    print 'Done with Cross-Validation (' + str(elapsedCV) + ' sec)'
     
    # Determine Optimal Parameters
    bestIndex = np.unravel_index(np.argmax(holdAcc), holdAcc.shape)
    bestC = C_list[bestIndex[0]]
    bestU = u_list[bestIndex[1]]
    bestCcon = Ccon_list[bestIndex[2]]
    print 'Best Parameters: C = ' + str(bestC) + ', u = ' + str(bestU) + ', Ccon = ' + str(bestCcon)
    
    # Evaluation using Optimal Parameters
    ctTrainFeats = np.concatenate([trainFeatsMod, trainFeatsRatMod/bestU])
    ctTrainLabels = np.concatenate([trainLabels, trainLabelsRat])
    W = np.concatenate([np.ones(trainLabels.shape), bestCcon*np.ones(trainLabelsRat.shape)])
       
    # Contrastive SVM
    modelRat = svm.SVC(C=bestC, kernel='linear', cache_size=2000, verbose=False)
    modelRat.fit(ctTrainFeats, ctTrainLabels, W)
    resultRat = modelRat.predict(testFeatsMod)
    ratAcc[splitID] = sum(resultRat==testLabels) / float(len(testLabels))
    
    # Regular SVM
    modelReg = svm.SVC(C=bestC, kernel='linear', cache_size=2000, verbose=False)
    modelReg.fit(trainFeats, trainLabels)
    resultReg = modelReg.predict(testFeats)
    regAcc[splitID] = sum(resultReg==testLabels) / float(len(testLabels))
    
    elapsedIter = (time.clock() - startIter)
    print 'Done with Iteration ' + str(splitID+1) + ' (' + str(elapsedIter) + ' sec)'
    print 'Contrastive SVM: ' + str(ratAcc[splitID])
    print 'Regular SVM: ' + str(regAcc[splitID]) + '\n'
    

# Mean Accuracy
ratAccMean = np.mean(ratAcc)
regAccMean = np.mean(regAcc)

print 'Final Results'
print 'Contrastive SVM (' + str(ratAccMean) +')'
print 'Regular SVM (' + str(regAccMean) +')'

# Save Results into CSV File
if savefile == 1:
    ratFilename = homedir + resultdir + 'result_ratCRF_' + word + '_sense' + str(targetSense) + '_split' + str(numSplits) + postfixReg + '.csv'
    np.savetxt(ratFilename, ratAcc, delimiter=",", fmt='%1.3f')
     
    regFilename = homedir + resultdir + 'result_regCRF_' + word + '_sense' + str(targetSense) + '_split' + str(numSplits) + postfixReg + '.csv'
    np.savetxt(regFilename, regAcc, delimiter=",", fmt='%1.3f')

