#coding=utf-8
from math import log
import operator

def createDataSet():
	"""测试阶段数据"""
	dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']	#数据集中的特征集合
	return dataSet,labels

def calcShannonEnt(dataSet):
	"""计算给定数据集的香农熵"""
	numEntries = len(dataSet)
	labelCounts = {}	#用于存储数据集中的类别
	for featVec in dataSet:
		currentLabel = featVec[-1] # 数据集最后一列为类别
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	# 计算香农熵,熵越高则混合的数据越多
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob,2)	#计算香农熵的公式
	return shannonEnt

# 待划分的数据集、划分数据集的特征、需要返回的特征的值
def splitDataSet(dataSet,axis,value):
	"""按照给定的特征axis划分数据集"""
	retDataSet = []
	# 将被用于划分数据集的特征从原数据集中剥离出来
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	"""在给定数据集中选取最好的特征用于分类"""
	numFeatures = len(dataSet[0])-1	#数据集中特征的数目，原始测试集中最后一列为类别
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		# 取得各个特征的取值集合 uniqueVals
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet)/float(len(dataSet))
			# 计算划分后的数据集的香农熵
			newEntropy += prob * calcShannonEnt(subDataSet)
		# 信息增益，即原始数据集的香农熵同根据某一特征划分后的数据集的香农熵的差值
		infoGain = baseEntropy - newEntropy
		# 取得使得信息增益最大的用于划分数据集的特征返回
		if(infoGain>bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	"""
	用于多数表决的代码，数据集已经取完了所有的属性，但是划分后的集合中的类别标签仍不唯一
	"""
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

def createTree(dataSet,labels):
	"""创建决策树的代码，递归的构建决策树"""
	classList = [example[-1] for example in dataSet]
	# 分支中的类别完全相同则停止继续划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 已经遍历完了所有用于分类的特征，通过多数表决返回类别
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	# 递归的构造决策树
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

# 决策树的分类函数
def classify(inputTree,featLabels,testVec):
	"""
		决策树，特征集合，测试向量
	"""
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key],featLabels,testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

# 使用pickle 模块存储决策树到硬盘上
def storeTree(inputTree,filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputTree,fw)
	fw.close()

# 从硬盘上获取数据
def grabTree(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)