dataArr, labelMat, = logRegres.loadDataSet()
weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
logRegres.plotBestFit(weights)
logRegres.multitest()