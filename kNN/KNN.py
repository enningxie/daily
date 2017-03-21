#coding=utf-8
from numpy import *
from os import listdir
import operator

def classify0(inX, dataSet, labels, k):
	"""
		分类器
		参数说明：用于分类的输入向量inX/输入的训练样本集dataSet/标签向量labels/用于选择最近邻向量的数目k
	"""
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet	#tile函数以inX为基准生成shape为(dataSetSize,1)大小的array
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()	#argsort()函数返回排序后的索引值
	classCount = {}	#用于投票表决的字典
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)	#将投票表决后的结果按逆序排序
	return sortedClassCount[0][0]	#返回类别

def img2vector(filename):
	"""
		将32x32存储的img转换成行向量
	"""
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(linStr[j])
	return returnVect

def handwritingClassTest():
	"""
		手写数字识别测试函数
	"""
	hwLabels = []
	trainingFileList = listdir('trainingDigits')	#将指定文件夹下的文件名以list的格式取出来
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])	#取出训练集中各个训练实例的标签
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
	testFileList = listdir('testDigits')	#取出测试集文件夹下的文件名
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print"The classifier came back with:%d,the real answer is :%d" % (classifierResult,classNumStr)
		if(classifierResult != classNumStr):
			errorCount += 1.0
	print"\nthe total number of errors is %d"%errorCount
	print"\nthe total error rate is %f" % (errorCount/float(mTest))