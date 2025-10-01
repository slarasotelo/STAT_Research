#install.packages("tree")
#install.packages("ISLR2")

library(tree)
library(ISLR2)
attach(Carseats)
High <- factor(ifelse(Sales <= 8, "No", "Yes")) #making it a binary variable

Carseats <- data.frame(Carseats, High) #merge high with data frame

tree.carseats <- tree(High ~. - Sales, Carseats) #fit classification tree to predict high using all variables except sale

summary(tree.carseats)

plot(tree.carseats)
text(tree.carseats, pretty = 0)

#training vs. test
set.seed(2)
train <- sample(1:nrow(Carseats), 200)
carseats.test <- Carseats[-train,]
high.test <- High[-train]
tree.carseats <- tree(High ~ . -Sales, Carseats, subset = train)
tree.pred <- predict(tree.carseats, carseats.test, type = "class")
table(tree.pred, high.test)

#PRUNING
set.seed(7)
cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)

par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b") #dev is the number of cross valisation 
plot(cv.carseats$k, cv.carseats$dev, type = "b")

prune.carseats <- prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)

tree.pred <- predict(prune.carseats, carseats.test, type = "class")
table(tree.pred, high.test)

prune.carseats <- prune.mmisclass(tree.carseats, best = 14)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
tree.pred <- predict(prune.carseats, carseats.test, type = "class")
table(tree.pred, high.test)

#fitting regression trees
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2) #training set
tree.boston <- tree(medv ~., Boston, subset = train)  #fitting tree
summary(tree.boston)

plot(tree.boston)
text(tree.boston, pretty = 0)

cv.boston <- cv.tree(tree.boston) #function that determine pruning the tree improves performance
plot(cv.boston$size, cv.boston$dev, type = "b") 

prune.boston <- prune.tree(tree.boston, best = 5) #pruning tree
plot(prune.boston) 
text(prune.boston, pretty = 0)

yhat <- predict(tree.boston, newdata = Boston[-train, ]) #use unpruned tree to make predictions
boston.test <- Boston[-train, "medv"] #boston test set
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test) ^ 2)

#bagging and random forests
#install.packages("randomForest")

library(randomForest)
set.seed(1)
bag.boston <- randomForest(medv ~., data = Boston, subset = train, mtry = 12, importance = TRUE) #bagging (where m = p), mtry = 12 says all 12 predictors should be considered
bag.boston

yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag - boston.test) ^ 2)


bag.boston <- randomForest(medv ~ . , data = Boston, 
                           subset = train, 
                           mtry = 12, 
                           ntree = 25)
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test) ^ 2)

#another random forest
set.seed(1)
rf.boston <- randomForest(medv ~., data = Boston, 
                          subset = train, 
                          mtry = 6, 
                          importance = TRUE)
yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test) ^ 2)

importance(rf.boston) #view importance of each variable
varImpPlot(rf.boston) #plots of importance measures


#Boosting

#install.packages("gbm")
library(gbm)
set.seed(1)
boost.boston <- gbm(medv ~., data = Boston[train, ], 
                    distribution = "gaussian", n.trees = 5000, 
                    interaction.depth = 4) #fit boosted regression trees
summary(boost.boston)

plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")

yhat.boost <- predict(boost.boston, newdata = Boston[-train, ], 
                      n.trees = 5000)
mean((yhat.boost - boston.test) ^ 2) #test MSE

boost.boston <- gbm(medv ~., data = Boston[-train, ], 
                    distribution = "gaussian", n.trees = 5000, 
                    interaction.depth = 4, shrinkage = 0.2, verbose = F)
yhat.boost <- predict(boost.boston, newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test) ^ 2)

#bayesian additive regression trees

#install.packages("BART")
library(BART)

x <- Boston[, 1:12]
y <- Boston[, "medv"]
xtrain <- x[train, ]
ytrain <- y[train]

xtest <- x[-train, ]
ytest <- y[-train]
set.seed(1)
barfit <- gbart(xtrain, ytrain, x.test = xtest) #fit bayesian additive regression tree model

yhat.bart <- barfit$yhat.test.mean
mean((ytest - yhat.bart) ^ 2) #test error

ord <- order(barfit$varcount.mean, decreasing = T)
barfit$varcount.mean[ord]

