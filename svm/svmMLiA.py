# coding=utf-8
from numpy import *


def loadDataSet(fileName):
    """加载数据集"""
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """在给定区间范围内随机选择一个整数
        第一个参数是alpha的下标，第二个参数是alpha的数目
    """
    j = i
    while(j==i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    """在数值太大的时候进行调整"""
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """简化版的SMO算法，输入参数分别是：数据集，类别标签、常数C、容错率和退出前最大的循环次数"""
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
            Ei = fxi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                    if L == H: print "L==H"; continue


