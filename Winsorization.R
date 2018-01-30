#Homework-1 (CS 731: ANN)
#Mohammed Farees Patel

library(ggplot2)
library(DescTools)
library(plyr)
library(neuralnet)
library(ggpubr)


#Download the Pima Diabetes dataset
dataset <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data", sep = ",",na.strings="0.0",strip.white=TRUE, fill = TRUE)

#removing all the NAs
pima <- dataset[complete.cases(dataset),] 

## changing the names of the attributes to more indentifiable names
names(pima) <- c("numpreg", "plasmacon", "bloodpress", "skinfold", "seruminsulin", "BMI", 
                 "pedigreefunction", "age", "classvariable")

colors<- c('red','blue','green','purple', 'aquamarine','cadetblue', 'chartreuse','cornflowerblue','chocolate1')

pima$classvariable<- as.numeric(pima$classvariable)

pima$classvariable<- ifelse(pima$classvariable==1,0,1)


#Creating a boxplot for all attributes
for(i in 1:(ncol(pima)-1)) { 
  nam <- paste("A", i, sep = "")
  assign(nam, ggboxplot(pima[,i], color=colors[i], xlab = names(pima)[i], ylab='Values'))
}

ggarrange(A1,A2,A3,A4,A5,A6,A7,A8, nrow=2, ncol=4)

#Load the dataset into a new variable
pima_std<- pima

#Perform z-standardization
cols<- c(1:8)
standardize <- function(x) as.numeric((x - mean(x)) / sd(x))
pima_std[cols] <- plyr::colwise(standardize)(pima_std[cols])

#Sampling 75% of the data as a training set and the remaining 25% as the test set
index <- sample(1:nrow(pima_std),round(0.75*nrow(pima_std)))
train <- pima_std[index,]
test <- pima_std[-index,]

#building the formulla to be used for the neural network
n<- names(pima_std)
f <- as.formula(paste("classvariable ~", paste(n[!n %in% "classvariable"], collapse = " + ")))

#Train the neural network
nn <- neuralnet(f,data=train,hidden=0, act.fct="logistic", linear.output=F, lifesign = "minimal")

#plot the neural network
plot(nn)


#Compute the training accuracy
pr.nn <- compute(nn, train[, 1:8])
pr.nn_ <- pr.nn$net.result
head(pr.nn_)

#training accuracy
mean(round(pr.nn_) == train[,9])

#Compute the testing accuracy
pr.test<- compute(nn,test[,1:8])
pr.test_<- pr.test$net.result
head(pr.test_)

#testing accuracy
mean(round(pr.test_) == test[,9])


#-------------------------------------------------------------------------------------------

#Load the dataset into a new variable
pima_win<- pima
Winsorizing <- function(x) as.numeric(Winsorize(x, probs = c(0.1,0.9)))
pima_win[cols] <- plyr::colwise(Winsorizing)(pima_win[cols])


#Creating a boxplot for all attributes
for(i in 1:(ncol(pima_win)-1)) { 
  nam <- paste("B", i, sep = "")
  assign(nam, ggboxplot(pima_win[,i], color=colors[i], xlab = names(pima_win)[i], ylab='Values'))
}

ggarrange(B1,B2,B3,B4,B5,B6,B7,B8, nrow=2, ncol=4)

#Perform z-standardization
pima_win_std<- pima_win
pima_win_std[cols] <- plyr::colwise(standardize)(pima_win[cols])

#Sampling 75% of the data as a training set and the remaining 25% as the test set
index_win <- sample(1:nrow(pima_win_std),round(0.75*nrow(pima_win_std)))
train_win_std <- pima_win_std[index,]
test_win_std <- pima_win_std[-index,]

#Building the formulla to be used for the neural network
n_win<- names(pima_win_std)
f_win <- as.formula(paste("classvariable ~", paste(n_win[!n_win %in% "classvariable"], collapse = " + ")))

#building the formulla to be used for the neural network
nn_win <- neuralnet(f_win,data=train_win_std,hidden=0, act.fct="logistic", linear.output=F, lifesign = "minimal")

#plot the neural network
plot(nn_win)


#Compute the training accuracy
pr.nn_win <- compute(nn_win, train_win_std[, 1:8])
pr.nn_win_result <- pr.nn$net.result
head(pr.nn_win_result)

#training accuracy
mean(round(pr.nn_win_result) == train_win_std[,9])

#Compute the testing accuracy
pr.test_win<- compute(nn_win,test_win_std[,1:8])
pr.test_win_result<- pr.test$net.result
head(pr.test_win_result)

#testing accuracy
mean(round(pr.test_win_result) == test_win_std[,9])


#Result
#The model performance over the observations inclusive of outliers had an accuracy of
#77% on the training data and 80% over the test data. Further, I trained my model on the
#winsorized training set and obtained an accuracy of about 81% on the testing set. As per
#the obtained results, I did not find much improvement in the model's performance over
#the winsorized data. Hence my conclusion depending on my analysis is that
#winsorization does not have a significant improvement in the performance of the model
#neither does decrease the model's performance.


#Conclusion
#As per my analysis, I inferred that for the Pima diabetes dataset the model performance
#does not significantly depend on winsorization. I believe that for winsorization to have
#any effect on the model performance it depends on the following characteristics:
#(i) The total number of outliers in the dataset
#(ii) Total number of attributes in the dataset
#(iii) Pre-determined lower and upper bounds for the extreme values
#(iv) Single-layer or Multi-layer perceptron

#End
