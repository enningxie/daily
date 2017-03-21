#coding=utf-8

from numpy import *

# 词表到向量的转换函数
def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],['maybe','not','take','him','to','dog','park','stupid'],['my','dalmation','is','so','cute','I','love','him'],['stop','posting','stupid','worthless','garbage'],['mr','licks','ate','my','steak','how','to','stop','him'],['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1] # 1代表侮辱性文字，0代表正常言论
	return postingList,classVec # 返回第一个变量是进行词条切分后的文档集合，第二个变量是一个类别标签的集合

# 会创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet|set(document) # 并集
	return list(vocabSet)

# 输入参数为词汇表及某个文档，输出的是文档向量
# 输出文档在词汇表中是否出现的标记向量
def bagofWords2VecMN(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

def setofWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
	return returnVec

# 朴素贝叶斯分类器训练函数
# 输入参数为文档矩阵trainMatrix，以及由每篇文档类别标签所构成的向量trainCategory
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)	#文档的数量
	numWords = len(trainMatrix[0]) 
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0Num = ones(numWords); p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯分类函数
# 第一个参数是要分类的向量
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify*p1Vec) + log(pClass1) # p1Vec就是经过log后的参数，此处的+相当于log中的值相乘
	p0 = sum(vec2Classify*p0Vec) + log(1.0-pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

# 操作整合函数
def testingNB():
	listOPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(                                                                                                                                                                                                                                (myVocabList,postinDoc))
	p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
	testEntry = ['love','my','dalmation']
	thisDoc = array(setofWords2Vec(myVocabList,testEntry))
	print testEntry,' classified as : ',classifyNB(thisDoc,p0V,p1V,pAb)
	testEntry1 = ['stupid','garbage']
	thisDoc1 = array(setofWords2Vec(myVocabList,testEntry1))
	print testEntry1,' classified as : ',classifyNB(thisDoc1,p0V,p1V,pAb)

# 接受一个大字符串并将其解析为字符串列表
def textParse(bigString):
	import re
	listOfTakens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTakens if len(tok) > 2]

# 对贝叶斯垃圾邮件分类器进行自动化处理
def spamTest():
	docList = []; classList = []; fullText = []
	for i in range(1,26): # 1-25
		wordList = textParse(open('spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	trainingSet = range(50); testSet = []
	# 构建测试集合
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setofWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	# 计算错误率
	errorCount = 0
	for docIndex in testSet:
		wordVector = setofWords2Vec(vocabList,docList[docIndex])
		if classifyNB(wordVector,p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
			print 'classification error ',docList[docIndex]
	print 'the error rate is: ',float(errorCount)/len(testSet)

# 遍历词汇表中的每个词统计它在文本中出现的次数
def calcMostFreq(vocabList,fullText):
	import operator
	freqDict = {} # freq 频率
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.iteritems(),key = operator.itemgetter(1),reverse=True)
	return sortedFreq[:30]

# 该函数使用两个Rss源作为参数，对测试进行自动化
def localWords(feed1,feed0):
	import feedparser
	docList = [];classList = []; fullText = []
	minLen = min(len(feed1['entries']),len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList,fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])
	trainingSet = range(2*minLen);testSet=[]
	for i in range(20):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = [];trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagofWords2VecMN(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagofWords2VecMN(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
			errorCount += 1
	print 'the error rate is : ',float(errorCount)/len(testSet)
	return vocabList,p0V,p1V

def getTopWords(ny,sf):
	import operator
	vocabList,p0V,p1V = localWords(ny,sf)
	topNY = []; topSF = []
	for i in range(len(p0V)):
		if p0V[i] > -6.0 : 
			topSF.append((vocabList[i],p0V[i]))
		if p1V[i] > -6.0 : 
			topNY.append((vocabList[i],p1V[i]))
	sortedSF = sorted(topSF,key=lambda pair: pair[1],reverse = True)
	print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF"
	for item in sortedSF:
		print item[0]
	sortedNY = sorted(topNY,key=lambda pair: pair[1],reverse = True)
	print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY"
	for item in sortedNY:
		print item[0]
