myDat, labels = trees.createDataSet()
myTree = trees.createTree(myDat, labels)
import treePlotter
treePlotter.createPlot(myTree)
myTree1 = trees.grabTree('lensesTree.txt')
treePlotter.createPlot(myTree1)