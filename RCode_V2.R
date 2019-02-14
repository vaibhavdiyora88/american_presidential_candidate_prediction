################################################################################
# Data Mining HW#3
################################################################################
# set memory limits
#options(java.parameters = "-Xmx64048m") # 64048 is 64 GB
#set.seed(1234)

# Connecting to a MySQL on the server
# Need to change 'lanhamm' to your user name. Make sure path classPath= is correct
#library(RJDBC)
#drv <- JDBC(driverClass="com.mysql.jdbc.Driver",classPath="/home/lanhamm/mysql-connector-java-5.1.47.jar")
#conn <- dbConnect(drv, url="jdbc:mysql://datamine.rcac.purdue.edu:3306/politics", user="gen_user", password="gen_user")

# run a SQL query and pull in data from the database
#h <- dbGetQuery(conn, "select * from household_data"
gc()
h <- read.csv("C:\\Users\\vaibh\\Desktop\\Fall\\data mining\\hw 3\\household_data.csv")
i <-  read.csv("C:\\Users\\vaibh\\Desktop\\Fall\\data mining\\hw 3\\individual_data.csv")
r <-  read.csv("C:\\Users\\vaibh\\Desktop\\Fall\\data mining\\hw 3\\registration_status.csv", sep = "|")

# examine structure of data

head(h)
head(i)
head(r)


# coerce variables to different data types
h$is_urban <- as.factor(h$is_urban)
h$is_owner_of_home <- as.factor(h$is_owner_of_home)
h$tercile_of_census_tract_income <- as.factor(h$tercile_of_census_tract_income)
i$is_head_of_household <- as.factor(i$is_head_of_household)
i$married <- as.factor(i$married)
i$race <- as.factor(i$race)

i$gender<- ifelse(i$gender == "Male" | i$gender == "MALE" | i$gender == "M", 1, 0)
i$gender <- as.factor(i$gender)
i$voted_in_2012 <- as.factor(i$voted_in_2012)
i$is_college_graduate <- as.factor(i$is_college_graduate)
r$is_registered_democrat <- as.factor(r$is_registered_democrat)
# distribution of response variable
table(r$is_registered_democrat)
# join the datasets into one (could also use SQL if you want)
library(plyr)
d <- join(x=r, y=i, by="person_id", type="inner")
head(d)
d <- join(x=d, y=h, by="hh_id", type="inner")

rm(h,i,r) # clean up environment

# re-arrange columns. Want target as first column
head(d)
d <- d[,c(2,1,3:14)]

# change column names
names(d)
names(d)[1] = "y"
names(d)[4] = "hoh"
names(d)[10] = "college_grad"
names(d)[13] = "home_owner"
names(d)[14] = "income_tercile"
names(d)

# change 'NA' to actually missing value NA; and make 1/2s as 0/1s
d$y <- ifelse(d$y == "NA", NA, d$y)
d$y <- ifelse(d$y == 1, 0, d$y)
d$y <- ifelse(d$y == 2, 1, d$y)
table(d$y)
d$y <- as.factor(d$y)
Y<-d$y
write.csv(d, 'd.csv')
d$y<-NULL
#creating dummy variables
Id<-d[,c(1,2)]
d[,c(1,2)]<-NULL
library(caret)
dummies <- dummyVars( ~., data = d)            # create dummyes for Xs
ex <- data.frame(predict(dummies, newdata = d))  # actually creates the dummies
names(ex) <- gsub("\\.", "", names(ex))          # removes dots from col names
head(ex)
ex[,c("hoh0", "marriedSingle", "gender0", "raceAsian","voted_in_20120", "college_grad0", "is_urban0", "home_owner0", "income_tercileBottom")] <- NULL
ex
d <- cbind(Id, ex)                              # combine your target variable with Xs
# make target variable called 'y'
rm(dummies, ex)    
head(d)
str(d)
############################################################################################
#preprocess
############################################################################################

str(d)
d<- cbind(Y,d)
# 'd' dataset and 'score_set'
score_set <- d[is.na(d$Y),]
d <- d[!is.na(d$Y),]
str(d)
str(score_set)
d[,2:3]<-NULL
Y <- d$Y
d$Y <- NULL

# calculate correlation matrix using Pearson's correlation formula
descrCor <-  cor(d[,3:ncol(d)])                           # correlation matrix
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .9) # num Xs with cor > t
summary(descrCor[upper.tri(descrCor)])                    # summarize the cors

# which columns in your correlation matrix have a correlation greater than some
# specified absolute cutoff. Find them and remove them
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.9)

# summarize those correlations to see if all features are now within our range
# update dataset by removing those filtered vars that were highly correlated
rm(descrCor,highCorr, highlyCorDescr)


library(caret)
# create a column of 1s. This will help identify all the right linear combos
d <- cbind(rep(1, nrow(d)), d[3:ncol(d)])
names(d)[1] <- "ones"

# identify the columns that are linear combos
comboInfo <- findLinearCombos(d)
comboInfo

# remove columns identified that led to linear combos
d <- d[, -comboInfo$remove]

# remove the "ones" column in the first column
d <- d[, c(2:ncol(d))]
rm(comboInfo) 

preProcValues <- preProcess(d[,2:ncol(d)], method = c("range"))
# Step 2) the predict() function actually does the transformation using the 
# parameters identified in the previous step. Weird that it uses predict() to do 
# this, but it does!
d <- predict(preProcValues, d)
head(d)
d<- cbind(Y, d)
levels(d$Y) <- make.names(levels(factor(d$Y)))
levels(d$Y)
table(d$Y)
# levels of a factor are re-ordered so that the level specified is first and 
# "X1" is what we are predicting. The X before the 1 has nothing to do with the
# X variables. It's just something weird with R. 'X1' is the same as 1 for the Y 
# variable and 'X0' is the same as 0 for the Y variable.
d$Y <- relevel(d$Y,"X0")
table(d$Y)
###########################################################################################
#data partition

set.seed(1234) # set a seed so you can replicate your results
library(caret)

# identify records that will be used in the training set. Here we are doing a
# 70/30 train-test split. You might modify this.
inTrain <- createDataPartition(y = d$Y,   # outcome variable
                               p = .70,   # % of training data you want
                               list = F)
# create your partitions
train <- d[inTrain,]  # training data set
test <- d[-inTrain,]  # test data set

# down-sampled training set
dnTrain <- downSample(x=train[,2:ncol(train)], y=train$Y)
###########################################################################################
# Specify cross-validation design
###########################################################################################
ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=5,        # k number of times to do k-fold
                     classProbs = T,  # if you want probabilities
                     summaryFunction = twoClassSummary, # for classification
                     #summaryFunction = defaultSummary,  # for regression
                     allowParallel=T)
ctrl
############################################################################################
#models
#############################################################################################
# train a logistic regession on train set 


myModel1 <- train(Y ~.,               # model specification
                  data = train,        # train set used to build model
                  method = "glm",      # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  family = "binomial", # specify the type of glm
                  metric = "ROC"       # performance measure
)
summary(myModel1)


# train a logistic regession on down-sampled train set 
#myModel2 <- train(Y ~ .,               # model specification
#                  data = dnTrain,        # train set used to build model
#                  method = "glm",      # type of model you want to build
#                  trControl = ctrl,    # how you want to learn
#                  family = "binomial", # specify the type of glm
#                  metric = "ROC"       # performance measure
#)
#myModel2

# train a feed-forward neural net on train set 
myModel3 <- train( Y~ .,               # model specification
                   data = train,        # train set used to build model
                   method = "nnet",     # type of model you want to build
                   trControl = ctrl,    # how you want to learn
                   tuneLength = 1:5,   # how many tuning parameter combos to try
                   maxit = 100,         # max # of iterations
                   metric = "ROC"       # performance measure
)
myModel3

# train a feed-forward neural net on the down-sampled train set using a customer
# tuning parameter grid
#myGrid <-  expand.grid(size = c(10,15,20)     # number of units in the hidden layer.
#                       , decay = c(.09,0.12))  #parameter for weight decay. Default 0.
#myModel4 <- train(Y ~ .,              # model specification
#                  data = dnTrain,       # train set used to build model
#                  method = "nnet",    # type of model you want to build
#                  trControl = ctrl,   # how you want to learn
#                  tuneGrid = myGrid,  # tuning parameter combos to try
#                  maxit = 100,        # max # of iterations
#                  metric = "ROC"      # performance measure
#)
#myModel4



# Capture the train and test estimated probabilities and predicted classes
# model 1 
logit1_trp <- predict(myModel1, newdata=train, type='prob')[,1]
logit1_trc <- predict(myModel1, newdata=train)
logit1_tep <- predict(myModel1, newdata=test, type='prob')[,1]
logit1_tec <- predict(myModel1, newdata=test)

# model 2 
#logit2_trp <- predict(myModel2, newdata=dnTrain, type='prob')[,1]
#logit2_trc <- predict(myModel2, newdata=dnTrain)
#logit2_tep <- predict(myModel2, newdata=test, type='prob')[,1]
#logit2_tec <- predict(myModel2, newdata=test)
# model 3
nn1_trp <- predict(myModel3, newdata=train, type='prob')[,1]
nn1_trc <- predict(myModel3, newdata=train)
nn1_tep <- predict(myModel3, newdata=test, type='prob')[,1]
nn1_tec <- predict(myModel3, newdata=test)
# model 4 
#nn2_trp <- predict(myModel4, newdata=dnTrain, type='prob')[,1]
#nn2_trc <- predict(myModel4, newdata=dnTrain)
#nn2_tep <- predict(myModel4, newdata=test, type='prob')[,1]
#nn2_tec <- predict(myModel4, newdata=test)

# train a C5.0 classification tree
myModel5 <- train(Y ~ .,               # model specification
                  data = train,        # train set used to build model
                  method = "C5.0Tree",    # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  tuneLength = 1:15,   # vary tree size
                  metric = "ROC"       # performance measure
)
myModel5

# train another type of classificationt tree: Boosted classification tree
myModel6 <- train(Y ~ .,               # model specification
                  data = train,        # train set used to build model
                  method = "ada",      # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  #tuneLength = 1:15,   # vary tree size
                  metric = "ROC"       # performance measure
)
myModel6

# Capture the train and test estimated probabilities and predicted classes
# model 5 
tree1_trp <- predict(myModel5, newdata=train, type='prob')[,1]
tree1_trc <- predict(myModel5, newdata=train)
tree1_tep <- predict(myModel5, newdata=test, type='prob')[,1]
tree1_tec <- predict(myModel5, newdata=test)
# model 6 
tree2_trp <- predict(myModel6, newdata=train, type='prob')[,1]
tree2_trc <- predict(myModel6, newdata=train)
tree2_tep <- predict(myModel6, newdata=test, type='prob')[,1]
tree2_tec <- predict(myModel6, newdata=test)



# model 1 - logit
(cm <- confusionMatrix(data=logit1_trc, train$Y))
(testCM <- confusionMatrix(data=logit1_tec, test$Y))
# model 2 - logit with down-sampled data
#(cm2 <- confusionMatrix(data=logit2_trc, dnTrain$Y))
#(testCM2 <- confusionMatrix(data=logit2_tec, test$Y))
# model 3 - nnet
(cm3 <- confusionMatrix(data=nn1_trc, train$Y))
(testCM3 <- confusionMatrix(data=nn1_tec, test$Y))
# model 4 - nnet with down-sampled data
#(cm4 <- confusionMatrix(data=nn2_trc, dnTrain$Y))
#(testCM4 <- confusionMatrix(data=nn2_tec, test$Y))
#tree models
#model 5
(cm5 <- confusionMatrix(data=tree1_trc, train$Y))
(testCM5 <- confusionMatrix(data=tree1_tec, test$Y))
# model 6 - another tree model
(cm6 <- confusionMatrix(data=tree2_trc, train$Y))
(testCM6 <- confusionMatrix(data=tree2_tec, test$Y))

###########################################################################################

################################################################################
# PROVIDE YOUR WORK BELOW
################################################################################

###########################################################################################

# create a training and testing. This one is 50%/50% train/test
# here I'll just identify the rows that will be used for train
rows = sample(1:nrow(d), round(nrow(d)*.5,0))
Y<-d$Y
d$Y<-NULL
head(d)
cost_df <- data.frame() #accumulator for cost results
cost_df
library(cluster)
for(k in 1:8){
  #allow up to 50 iterations to obtain convergence, and do 20 random starts
  # train set
  # column 6th and 9th are numerical varaible.
  kmeans_tr <- kmeans(x=d[rows, c(4,7)], centers=k, nstart=20, iter.max=50)
  # test set
  kmeans_te <- kmeans(x=d[-rows, c(4,7)], centers=k, nstart=20, iter.max=50)
  
  #Combine cluster number and cost together, write to df
  cost_df <- rbind(cost_df, cbind(k, kmeans_tr$tot.withinss
                                  , kmeans_te$tot.withinss))
}

#elbow plot
names(cost_df) <- c("cluster", "tr_cost", "te_cost")
cost_df

par(mfrow=c(1,1))
cost_df[,2:3] <- cost_df[,2:3]/100
plot(x=cost_df$cluster, y=cost_df$tr_cost, main="k-Means Elbow Plot"
     , col="blue", pch=19, type="b", cex.lab=1.2
     , xlab="Number of Clusters", ylab="MSE (in 10^2s)")
points(x=cost_df$cluster, y=cost_df$te_cost, col="green")

################################################################################
# Looking at the information from your k-means clustering
# Lets just generate a k-means clustering for k=2
################################################################################
kmeans_tr <- kmeans(x=d[rows, c(4,7)], centers=2, nstart=20, iter.max=50)
kmeans_te <- kmeans(x=d[-rows,c(4,7)], centers=2, nstart=20, iter.max=50)

# You can look at the fit object to to see how many records are in each cluster,
# the final centroids, the final cluster assignments, statistics of within and
# between clusters
kmeans_tr

# get cluster means (couple different ways)
(centroids <- aggregate(d[rows, c(4,7)], by=list(kmeans_tr$cluster), FUN=mean))
kmeans_te$centers

# see the number of obs within each cluster
kmeans_tr$size 
kmeans_te$size

################################################################################
# Silhouette values
library(cluster)
# kmeans (k=3)
km3 <- kmeans(d[,c(4,7)], 3)
dist3 <- dist(d[,c(4,7)], method="euclidean")
sil3 <- silhouette(km3$cluster, dist3)
plot(sil3, col=c("black","red","green"), main="Silhouette plot (k=3) K-means")

# kmeans (k=2)
km2 <- kmeans(d[,c(4,7)], 2)
dist2 <- dist(d[,c(4,7)], method="euclidean")
sil2 <- silhouette(km2$cluster, dist2)
plot(sil2, col=c("black","red"), main="Silhouette plot (k=2) K-means")

# kmeans (k=4)
km4 <- kmeans(d[,c(4,7)], 4)
dist4 <- dist(d[,c(4,7)], method="euclidean")
sil4 <- silhouette(km4$cluster, dist4)
plot(sil4, col=c("black","red", "blue", "green"), main="Silhouette plot (k=4) K-means")


######################################################################################
#Cluster validation
######################################################################################
# create a training and testing set
set.seed(1234)
rows = sample(1:nrow(d), round(nrow(d)*.7,0))
train = d[rows, ]
test = d[-rows, ]
# perform k-means on training set
library(flexclust)
km2 <- kcca(x=train[,c(4,7)], k=2, family=kccaFamily("kmeans"))
km3 <- kcca(x=train[,c(4,7)], k=3, family=kccaFamily("kmeans"))
#### OPTION 1 ####
# predict cluster assignment using our build models (km2 and km3) using data
# from the test set
clust2 <- predict(km2, test[,c(4,7)])
clust3 <- predict(km3, test[,c(4,7)])
test$clust2 <- clust2
test$clust3 <- clust3
# average silhouette values by cluster on test data set
# k=2
km2_train <- data.frame(rep("train k=2",nrow(km2@"centers"))
                        ,cbind(c(1:nrow(km2@"centers")), km2@"centers"))
km2_test <- data.frame(rep("test k=2",nrow(km2@"centers")),
                       aggregate(test[,c(4,7)], by=list(test[,"clust2"]), FUN=mean))
names(km2_train)[1:2] <- c("Dataset","Cluster")
names(km2_test)[1:2] <- c("Dataset","Cluster")
km2_test
# k=3
km3_train <- data.frame(rep("train k=3",nrow(km3@"centers"))
                        ,cbind(c(1:nrow(km3@"centers")), km3@"centers"))
km3_test <- data.frame(rep("test k=3",nrow(km3@"centers")),
                       aggregate(test[,c(4,7)], by=list(test[,"clust3"]), FUN=mean))
names(km3_train)[1:2] <- c("Dataset","Cluster")
names(km3_test)[1:2] <- c("Dataset","Cluster")
# all results merged together - want this table to compare train and test stats
# for each model and cluster
results <- rbind( km2_train, km2_test,km3_train, km3_test)
results


###########################################################################################
# train Control
# some more models
###########################################################################################
set.seed(1234) # set a seed so you can replicate your results
library(caret)
d<-cbind(Y,d)
# identify records that will be used in the training set. Here we are doing a
# 70/30 train-test split. You might modify this.
inTrain <- createDataPartition(y = d$Y,   # outcome variable
                               p = .70,   # % of training data you want
                               list = F)
# create your partitions
train <- d[inTrain,]  # training data set
test <- d[-inTrain,]
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 22)

## For the last model:
seeds[[51]] <- sample.int(1000, 1)

ctrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     seeds = seeds)
mod <- train(Y ~ ., data = train,
             method = "knn",
             tuneLength = 12,
             trControl = ctrl)
mod
install.packages("ISLR")
library(ISLR)
mod1 <- glm(Y ~., data=d, family=binomial)
mod1

model7_trp <- predict(mod, newdata=train, type='prob')[,1]
model7_trc <- predict(mod, newdata=train)
model7_tep <- predict(mod, newdata=test, type='prob')[,1]
model7_tec <- predict(mod, newdata=test)



model8_trc <- predict(mod1, newdata=train)
model8_tec <- predict(mod1, newdata=test)
(cm7 <- confusionMatrix(data=model7_trc, train$Y))
(testCM7 <- confusionMatrix(data=model7_tec, test$Y))
#(cm8 <- confusionMatrix(data=model8_trc, train$Y))
#(testCM8 <- confusionMatrix(data=model8_tec, test$Y))

#Random forest method
ctrl1 <- trainControl(method="cv",     # cross-validation set approach to use
                      number=3,        # k number of times to do k-fold
                      classProbs = T,  
                      summaryFunction = defaultSummary,
                      allowParallel=T)

rf <- train(Y ~ .,
            data = train,
            method = "rf",
            importance=T,    # we add this in or it varImp cannot be computed
            trControl = ctrl1,
            tuneLength = 10,
            metric = "ROC")
rf

model9_trp <- predict(rf, newdata=train, type='prob')[,1]
model9_trc <- predict(rf, newdata=train)
model9_tep <- predict(rf, newdata=test, type='prob')[,1]
model9_tec <- predict(rf, newdata=test)

(cm9 <- confusionMatrix(data=model9_trc, train$Y))
(testCM9 <- confusionMatrix(data=model9_tec, test$Y))
#########################################################################################
# calculating mccr for all models
########################################################################################
act<- train$Y
act_test<-test$Y
act<- ifelse(act=="X0", 0, 1)
act_test<- ifelse(act_test=="X0", 0, 1)
pred1<-ifelse(logit1_trc=="X1", 1,0)
pred12<-ifelse(logit1_tec=="X1", 1,0)
pred3<-ifelse(nn1_trc=="X1", 1,0)
pred32<-ifelse(nn1_tec=="X1", 1,0)
pred5<-ifelse(tree1_trc=="X1", 1,0)
pred52<-ifelse(tree1_tec=="X1", 1,0)
pred7<-ifelse(model7_trc=="X1", 1,0)
pred72<-ifelse(model7_tec=="X1", 1,0)
pred9<-ifelse(model9_trc=="X1", 1,0)
pred92<-ifelse(model9_tec=="X1", 1,0)
table(act_test)
table(pred1)

#model1
(mccr(act, pred1))
(mccr(act_test, pred12))
#model3
(mccr(act, pred3))
(mccr(act_test, pred32))
#model5
(mccr(act, pred5))
(mccr(act_test, pred52))
#model7
(mccr(act, pred7))
(mccr(act_test, pred72))
#model9
(mccr(act, pred9))
(mccr(act_test, pred92))

###########################################################################################
# automatic backward selection
##########################################################################################
library(leaps)
mb <- regsubsets(Y ~ ., data=train
                 , nbest=1
                 , intercept=T
                 , method='backward'
                 , really.big=T
)

# You can pick the model you want based on the performance measure you specify. 
# Here I used 'adjr2', but you could select based on 'cp', 'bic', 'rss'
vars2keep <- data.frame(summary(mb)$which[which.max(summary(mb)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]

# here are the final features found to be statistically significant
vars2keep

model_formula <- paste(" Y ~ hoh1 + marriedMarried + raceBlack + age + college_grad1 + 
                       is_urban1+ income_tercileMiddle+ income_tercileTop")

myModel1 <- train( Y ~ hoh1 + marriedMarried + raceBlack + age + college_grad1 + 
                     is_urban1+ income_tercileMiddle+ income_tercileTop,               # model specification
                   data = train,        # train set used to build model
                   method = "glm",      # type of model you want to build
                   trControl = ctrl1,    # how you want to learn
                   family = "binomial", # specify the type of glm
                   metric = "ROC"       # performance measure
)
summary(myModel1)

##########################################################################################
#area under the curve auc
##########################################################################################
install.packages("pROC")
library(pROC)
roc1 <- roc(response=act, predictor = pred1)
roc5 <- roc(response=act, predictor = pred5)
auc(roc1)
auc(roc5)
par(mfrow=c(1,1))
plot(roc1, main="ROC curves", legacy.axes =T, col="red")
lines(roc5,col="green")
legend("bottomright", inset=0, title="models", legend = c("logit","tree c5.0"), 
       fill=c("red","green"))
###########################################################################################
#Final model used
###########################################################################################
final_model <- train(Y ~ .,
            data = d,
            method = "rf",
            importance=T,    # we add this in or it varImp cannot be computed
            trControl = ctrl1,
            tuneLength = 10,
            metric = "ROC")
final_model

model10_trp <- predict(final_model, newdata=score_set, type='prob')[,1]
model10_trc <- predict(final_model, newdata=d)

(cm10 <- confusionMatrix(data=model10_trc, d$Y))

head(model10_trp)
table(model10_trc)
table(d$Y)
head(score_set)

act_final<- ifelse(d$Y=="X0", 0, 1)
pred_final<-ifelse(model10_trc=="X1", 1,0)
(mccr(act_final, pred_final))

vaibhav_scores <- cbind(score_set$person_id, model10_trp)
vaibhav_scores <- as.data.frame(vaibhav_scores)
names(vaibhav_scores) = c("person_id", "scores")
head(vaibhav_scores)

write.csv(vaibhav_scores, 'vaibhav_scores.csv')
