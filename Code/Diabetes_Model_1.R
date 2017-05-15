###############################################################################
#
# Project: 	Capstone
# Script:	Diabetes_Model_1.R
# Version:	01
# Created:	Oct 12, 2016
# Updated:	Oct 13, 2016
# Author: 	Robert P. Yerex
# Copyright Robert P. Yerex, 2016
#
###############################################################################

#	Required Libraries

library(caret)
library(pROC)
library(plyr)
library(data.table)
library(gbm)
library(ada)


dataSet <- fread("Data/DMReadmitDataInput.csv", header = TRUE)

#---------------------------------------------------------------------------------------------
#	In the original data set, the variable readmitted has values: {NO, <30, >30}. In order to
#	use this as a categorical target, it is necessary to transform it. The created variables
#	are: readmitted_LT_30, readmitted_GT_30, and dataSet$readmitted_Any.
#	These are created as logical variables by default. 
#---------------------------------------------------------------------------------------------

dataSet$readmitted_LT_30 <- dataSet$readmitted == '<30'
dataSet$readmitted_GT_30 <- dataSet$readmitted == '>30'
dataSet$readmitted_Any <- dataSet$readmitted_LT_30 | dataSet$readmitted_GT_30

#---------------------------------------------------------------------------------------------
#	Most of the R modeling algorithms require that character variables be transformed into factors.
#	Rather than doing this one at a time, the code below uses the R function "lapply" to transform
#	all the variable specified in the list "cols". The target variables readmitted, readmitted_LT_30,
#	readmitted_GT_30, and readmitted_Any are also changed to factors because some model algorithms,
#	including "ada", do not work properly when the target is a logical variable
#---------------------------------------------------------------------------------------------

cols <- c('race','gender','admission_type_id','discharge_disposition_id','admission_source_id','payer_code',
		'medical_specialty','diag_1','diag_2','diag_3','max_glu_serum','A1Cresult','change','diabetesMed',
		'readmitted','readmitted_LT_30','readmitted_GT_30','readmitted_Any')  

dataSet[,(cols):=lapply(.SD, as.factor),.SDcols=cols]

#---------------------------------------------------------------------------------------------
#	This is a large data set, so for initial testing purposes, the following code creates a
#	small none random subset of the first 500. This value can be changed, or when ready to
#	use the full data set, comment out the line below and replace with "readmitData <- dataSet
#---------------------------------------------------------------------------------------------
readmitData <- dataSet[1:500,]

#---------------------------------------------------------------------------------------------
#	The code below creates the data frames that will be used in running the model:
#
#		modeldata 	includes only the variables needed
#		features	specifies the features, or RHS variables
#		target		specifies the LHS or Y variable
#
#	Setting up the features and target data frames allows for simplified specification of the 
#	model equation
#---------------------------------------------------------------------------------------------
modeldata <- as.data.frame(na.omit(subset(readmitData, select = c("race","gender","age","admission_type_id",
								"discharge_disposition_id","admission_source_id","num_procedures","num_medications",
								"number_inpatient","diag_1","diag_2","diag_3","number_diagnoses","readmitted_LT_30"))))

features <- as.data.frame(subset(modeldata, select = c("race","gender","age","admission_type_id",
						"discharge_disposition_id","admission_source_id","num_procedures",
						"num_medications","number_inpatient","number_diagnoses","diag_1",
						"diag_2","diag_3")))

target <- modeldata$readmitted_LT_30

#-----------------------------------------------------------------------------------------------
#	As with most stochastic processes in R, setting the "seed" allows for replication of results
#-----------------------------------------------------------------------------------------------
set.seed(1)

#-------------------------------------
# Setup traing and test partitions
#-------------------------------------
inTrain <- createDataPartition(target, p = 3/4, list = FALSE)

trainFeatures <- features[inTrain,]
testFeatures  <- features[-inTrain,]
trainTarget <- target[inTrain]
testTarget  <- target[-inTrain]
prop.table(table(target))
prop.table(table(trainTarget))
ncol(trainFeatures)

#----------------------------
#	Test feature distribution
#----------------------------
nearZeroVar(features, saveMetric = TRUE)

#---------------------------------------------------------------------------------------------
# Model Creation & Evaluation
#------------------------------
#	The "caret" package provides a "training harness", a way to test multiple combinations of
#	tuning parameters. In the code below, "adaGrid" is a data frame contining all combinations
#	of the "iter" and "maxdepth" vectors. If this grid is large, computing time will be long
#	The commented line specifies a large grid while the line below it is a smaller grid
#	for initial testing purposes
#---------------------------------------------------------------------------------------------
fitControl <- trainControl(method = "boot632", number = 10)
#adaGrid <- expand.grid(iter = c(10,50,150,200), maxdepth = c(1,2,3,5,10), nu = 1)
adaGrid <- expand.grid(iter = c(5,10), maxdepth = c(1,5), nu = 1)

adaFit <- train(trainFeatures, trainTarget,
		method = "ada",
		trControl = fitControl,
		tuneGrid = adaGrid,
		verbose = TRUE)

adaFit											# Display model results
#adaFit$finalModel								# Provides details of the final model

#---------------------------------------------------------------------------------------------
#	This section of code generates a set of graphs useful for diagnosing the model fitting
#	process. The call to "x11()" simply starts a new graphics window/page. Without it, each
#	successive graph would overwrite the previous one. There are many other, more elegant
#	ways to handle multiple graphs which can be found in the documetnation for "ggplot"
#---------------------------------------------------------------------------------------------
x11()
plot(adaFit)									# Plot of accuracy as a function of # trees and max tree depth
x11()
plot(adaFit$finalModel,TRUE,TRUE)				# Plot training error and Kappa as a function of interations
x11()
resampleHist(adaFit)							# Plot Accuracy and Kappa distributions from bootstrapped resampling 
x11()
dotPlot(varImp(adaFit))							# Plot of variable importance


#---------------------------------------------------------------------------------------------
#	This section of code gathers predicted and observed values from the test data set using
#	the best fit model on the training data to score. It then computes the confusion matrix
#---------------------------------------------------------------------------------------------
models <- list(ada = adaFit)
testPred <- predict(models, newdata = testFeatures)
predValues <- extractPrediction(models, testX = testFeatures, testY = testTarget)
testValues <- subset(predValues, dataType == "Test")
head(testValues)
table(testValues$model)
nrow(testFeatures)
adaConfusionMatrix <- confusionMatrix(testValues$pred, testValues$obs)
adaConfusionMatrix

#-----------------------------------------------------------
#	Extract probabilities and generate a ROC or Recall curve
#-----------------------------------------------------------
probValues <- extractProb(models,testX = testFeatures, testY = testTarget)
testProbs <- subset(probValues,	dataType == "Test")
str(testProbs)
x11()
adaROC <- roc(testProbs$obs, testProbs$FALSE., plot = TRUE)
