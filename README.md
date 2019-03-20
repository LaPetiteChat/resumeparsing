# resumeparsing
简历过滤器的ML模型
首先，简历过滤需要一位数据处理专家对数据进行预处理，预处理的过程中，将申请人的degree，education，skills进行赋值处理。然后将所有数据分成训练集、测试集，并建立模型进行计算。计算出的结果，进行评分，并按照评分结果进行排名。分别按照：
第一步：数据的处理；
第二步：数据的分类汇总；
第三步：做出预测；
第四步：预测的AB测试；
第五步：预测的准确度；
第六步：汇总所有的codes并执行。


我参考的是James Brownlee在纯数据计算朴素贝叶斯模型的python代码。https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

数据处理的结果应该按照是如下格式：
2,80,75,77.5,170,76.5623006,100,0.477837813,0.48099608;
1,60,60,60,170,62.56563666,100,0.41483181,0.018300161;
1,70,60,65,180,11.8019322,100,0.587375064,0.59734792;
2,95,90,92.5,180,9.192119736,100,0.724807147,0.793548513;
2,70,60,65,190,61.05618588,100,0.210516292,0.847069275;
2,80,60,70,175,53.8670648,100,0.611277028,0.548198959;
2,98,80,89,180,97.73447261,100,0.327901412,0.901802248;
2,100,70,85,190,15.40299551,100,0.102116189,0.737199743;
2,80,60,70,170,19.10572653,100,0.304586768,0.990202928;
2,75,80,77.5,175,56.2798471,100,0.188033658,0.031831499;
2,60,60,60,108,66.80843217,100,0.31256692,0.721842211;
1,60,60,60,128,97.83551224,100,0.31916056,0.874652191;
1,60,60,60,118,87.06427613,100,0.60115086,0.350288914;
1,80,60,70,148,18.23221236,100,0.203243346,0.647979462;
1,70,60,65,96,92.66992657,100,0.636451279,0.674574266;
2,70,60,65,56,30.32381303,100,0.052958132,0.367499219;
2,70,60,65,78,97.02955264,100,0.838578477,0.237851524;
1,75,60,67.5,99,43.84248184,100,0.429382186,0.963473002;
1,80,60,70,80,16.93151228,100,0.590318547,0.612625236;
2,89,60,74.5,76,68.57730357,100,0.812078797,0.023798746;
2,75,60,67.5,70,34.40610591,100,0.798953348,0.747503739;
2,80,60,70,70,68.67798479,100,0.594131327,0.35679144;
1,80,65,72.5,70,42.83178707,100,0.20246207,0.759385263;
2,80,70,75,78,94.55595774,100,0.989418371,0.719005387;
2,80,70,75,80,31.86938847,100,0.011526203,0.93627719;
2,95,80,87.5,80,33.70595686,100,0.416061459,0.396876282;
2,80,70,75,80,32.24077154,100,0.306767368,0.380221452;
2,75,70,72.5,70,73.98029002,100,0.869762403,0.36895025;
2,95,80,87.5,75,42.24168725,100,0.145999377,0.84569649;
2,70,60,65,65,21.85085821,100,0.433577036,0.139770985;
2,80,60,70,70,94.27238981,100,0.272838291,0.514772471;
2,80,80,80,180,50.52999248,100,0.478211656,0.115011178;
2,80,60,70,70,28.98835362,100,0.337907792,0.163271458;
2,70,70,70,70,42.22976876,100,0.923559771,0.159577817;
2,95,80,87.5,88,99.49893906,100,0.629810507,0.240372341;
2,70,60,65,60,58.33858912,100,0.532191584,0.311663668;
2,75,60,67.5,65,37.64067938,100,0.017666922,0.123165325;
2,95,70,82.5,76,44.28547402,100,0.614640793,0.57501755;
2,80,60,70,75,55.65959732,100,0.905169823,0.959682442;
4,80,70,75,70,4.119718261,100,0.131295751,0.838565154;
1,95,80,87.5,88,62.25351263,100,0.046046188,0.749358644;
2,95,70,82.5,138,11.04213716,100,0.830072134,0.261641346;
1,80,70,75,148,22.23522883,100,0.951078931,0.112332275;
2,75,60,67.5,118,94.81548956,100,0.090262754,0.968252204;
1,75,60,67.5,108,45.05536366,100,0.841871612,0.846410278;
2,80,70,75,180,19.3636175,70,0.855828096,0.759138723;
2,85,70,77.5,100,60.06062389,70,0.131260472,0.110782428;
2,80,60,70,180,9.138379193,70,0.766458516,0.222152779;
1,60,60,60,180,40.79941226,70,0.853640842,0.491858233;
1,70,60,65,118,61.37785318,70,0.685999587,0.854444086;
2,85,70,77.5,178,94.46345188,70,0.066474014,0.352098625;
1,80,70,75,100,95.4459327,70,0.285268038,0.025587163;
2,95,80,87.5,100,15.66591961,70,0.790453718,0.254196496;
2,60,60,60,128,0.44465934,70,0.398131268,0.069557991;
1,80,70,75,118,19.29234555,70,0.704691473,0.063151475;
4,80,70,75,148,49.61073014,70,0.156932286,0.754304454;
1,75,70,72.5,180,57.61196301,70,0.934810363,0.315597648;
2,75,70,72.5,138,7.793499915,70,0.48824784,0.987923607;
2,80,70,75,128,61.27801532,70,0.073013419,0.989033413;
1,75,60,67.5,138,73.70227221,70,0.188176053,0.127624627;
2,70,60,65,180,51.67259957,70,0.31772058,0.103774004;
2,80,70,75,178,36.12180926,70,0.316417386,0.846812165;
1,75,60,67.5,128,91.51184891,70,0.775280982,0.268165881;
2,95,80,87.5,128,50.04427752,70,0.093976794,0.326069983;
2,80,70,75,138,84.97538663,80,0.029917753,0.760080103;
2,75,60,67.5,148,65.63419798,80,0.537541227,0.781482787;
1,80,70,75,130,55.12483937,80,0.24535984,0.508181576;
1,80,70,75,130,83.15918437,80,0.132191626,0.45038017;
2,80,70,75,128,10.61381313,80,0.566090299,0.688339999;
2,80,70,75,128,99.07948967,80,0.769833272,0.108526708;
2,70,60,65,130,85.71394939,80,0.494068544,0.548402184;
2,75,60,67.5,148,47.73810582,80,0.624882073,0.421783056;

假设：Degree: Bachelor=1, Master=2, Phd=4
      Education1: 各高效排名参照国内高校排名赋值，权重为50%
Education2:对各高校计算机专业排名综合评分表排名, Fudan=80,Shanghai Jiao Tong=95, Nanjing=80，…，权重为50%
Education3:按照1：1比例计算出一个education的综合分
      Skills里面分为：C++=100,Java=90,Python=98,SQL=80, …
      Work Experience: ibm=80,Huawei=90,Microsoft=70,…
      Position: dev为100， qa=80, manager=70, no hire=50
      Years of Work Experience:
      按照技能分隔符的个数赋值：比如一个技能赋值为1，两个技能赋值为2，以此类推等等。
      Plus: 按照关键技能的关键词出现频率排名，其中debug为5，dev为4，program为4，test为3，…
（以上数据都是假设，具体计算的时候按照具体设计来进行，为了节约时间，我自动生成了一些随机的0到1的数字来补充空白的数字。）
这个数据库一共提供了72个记录，就是72个候选人的数据。
如果有no hire,可以赋值null数据为50。
数据集分类方法：
数据集可以按照67%训练集对33%测试的比例分配，在预测的时候将数据分为5份，并检验其数据的结果如何。


所有代码如下：
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
 
def loadCsv(employeesdata1.csv)
	lines = csv.reader(open(employeesdata1.csv, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
def main():
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)
 
main()

最后运行汇总的结果应该如下：
1.	Split 72 rows into train=48 and test=24 rows
2.	Accuracy: 76.3779527559%

