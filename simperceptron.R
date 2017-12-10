# Simple Perceptron 
# Student Admission Prediction in Texas Tech

# total score: 2400
SATScore<- c(1100, 1200, 800, 2300, 1450, 900, 800, 1700, 1500, 2100)
GPA <-     c(2.5, 3.5,  3, 4, 3.75, 3.9, 2, 3.5, 3, 3.5)
# Letter of Recommendation 1 is the lowest 4 is the best
LoR <-      c(2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0, 3.0, 3.0,  4.0)
Admitted <- c(0, 1, 0, 1, 1, 0, 0, 1, 0, 1)

# simple perception model design: i.e. also called 
# single-layer Neural Networks

df <- data.frame(SATScore, GPA, LoR, Admitted)
df
str(df)

# Use normalization technique to categorize the number
# between 0 to 1, also called min-max normalization technique in order
# to follow the neural network format. In this case admitted has 1 or 0
# rest we need to convert. 

df$SATScore <- (df$SATScore - min(df$SATScore))/(max(df$SATScore) - min(df$SATScore))
df$GPA <- (df$GPA - min(df$GPA))/(max(df$GPA) - min(df$GPA))
df$LoR <- (df$LoR - min(df$LoR))/(max(df$LoR) - min(df$LoR))

# Use set.seed for random number generator, use ?set.seed for more info in Rstudio
# categorize the data into test and training

#set.seed(222)

takeSam <- sample(2, nrow(df), replace = TRUE, prob = c(0.8, 0.2))

training <- df[takeSam == 1, ]
test <- df[takeSam == 2, ]

# install package neuralnet. 

library(neuralnet)

nn <- neuralnet(Admitted~SATScore+GPA+LoR, data = training,
                hidden = 4,
                err.fct = 'ce',
                linear.output = FALSE )

plot(nn)
# prediction 

## Remove the admitted category/row data 
output1 <- compute(nn, training[, -4])


# prediction model: 

predmodel <- output1$net.result

prediction <- ifelse(predmodel>0.5, 1, 0)

# Make a data in a tabular form

table1 <- table(predmodel, training$Admitted)
table1
str(table1)

# check the misclassification error - training data

1-sum(diag(table1))/sum(table1)

# check the misclassification error - training data
output1 <- compute(nn, test[, -4])


# prediction model: 

predmodel1 <- output1$net.result

prediction1 <- ifelse(predmodel1>0.5, 1, 0)

# Make a data in a tabular form

table2 <- table(predmodel1, test$Admitted)
table2

# check the misclassification error - training data

1-sum(diag(table2))/sum(table2)

