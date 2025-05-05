#######################################
##REGULARIZED (L2 - RIDGE) REGRESSION##
#######W/ PSEUDOINVERSE SOLUTION#######
#######################################

#LOAD LIBRARIES
library(tidyverse)

#IMPORT DATA
df<-read.table('https://raw.githubusercontent.com/slevkoff/CLASS_DATA/master/marketing_bivariate.txt')

#VIEW THE DATA IN SPREADSHEET VIEW
View(df)

#PLOT THE DATA USING BASE SYSTEM
ggplot(df, aes(x=ad_time, y=units_sold)) +
  geom_point() +
  theme_classic()

#MANUALLY ADDING NONLINEAR (POLYNOMIAL) FEATURE TRANSFORMATION(S)
df$ad_time2 <- df$ad_time^2 #QUADRATIC TRANSFORMATION (2nd ORDER)
df$ad_time3 <- df$ad_time^3 #CUBIC TRANSFORMATION (3rd ORDER)

#VIEW AGAIN TO CONFIRM "MANUAL" ADDITION OF TRANFORMATION COLUMNS
View(df)

#PARTITIONING THE DATA

#CREATE A VARIABLE TO REFERENCE RANDOM ROWS TO PULL FOR TRAINING DATA
set.seed(10) #SAME SEED AS LAST CODE CHUNK FOR COMPARABILITY OF RESULTS
training_rows <- runif(dim(df)[1])>.3 #RANDOM VARIABLE THAT IS TRUE 70% OF TIME
training <- df[training_rows,] #PULL TRAINING ROWS
holdout <-df[!training_rows,] #PULL NON TRAINING ROWS

#########################################
#IMPLEMENTING THE PSEUDOINVERSE SOLUTION#
#########################################

#STEP 1: FORM THE INPUT MATRIX X:

#STEP 1.1: MAKE A COLUMN OF ONES TO INCLUDE AS REGRESSORS FOR INTERCEPT
col_of_ones <- rep(1, dim(training)[1])

#STEP 1.2: BIND COLUMN OF ONES WITH OTHER INPUT DATA COLUMNS
#AND COERCE TO MATRIX OBJECT
X <- as.matrix(cbind(col_of_ones, training[,-1]))

#STEP 2: FORM THE OUTPUT VECTOR y
y <- training[,1]

#STEP 3: COMPUTE THE PSEUDOINVERSE MATRIX
X_pseudo <- solve(t(X)%*%X)%*%t(X)

#STEP 4: MULTIPLY THE PSEUDOINVERSE MATRIX BY THE OUTPUT VECTOR
Betas <- X_pseudo%*%y

#REPORT OPTIMAL COEFFICIENTS
Betas

#CONFIRM WITH lm() FUNCTION
lm(units_sold ~ ad_time + ad_time2 + ad_time3, training)$coefficients

#GENERATE IN-SAMPLE PREDICTIONS ON TRAINING PARTITION
PRED_IN <- X%*%Betas

#CREATE INPUT MATRIX FOR HOLDOUT PARTITION
X_holdout <- as.matrix(cbind(col_of_ones[1:dim(holdout)[1]], holdout[,-1]))

PRED_OUT <- X_holdout%*%Betas

y_holdout <- df[!training_rows,1]

#COMPUTING IN AND OUT OF SAMPLE RMSE
(RMSE_IN <- sqrt(mean((y-PRED_IN)^2)))
(RMSE_OUT <- sqrt(mean((y_holdout-PRED_OUT)^2)))

#############################
#IMPLEMENTING REGULARIZATION#
#############################

#LET'S IMPLEMENT SOME REGULARIZATION USING THE L2 RIDGE PENALTY

#SET UP GRID OF REGULARIZATION PARAMETER (LAMBDA) VALUES
lambda <- seq(0, 2,.001)

#INITIALIZE EMPTY MATRIX TO STORE ESTIMATED MODEL COEFFICIENTS FOR EACH LAMBDA
BETA_RIDGE <- matrix(NA, nrow = dim(t(X)%*%X)[1], ncol=length(lambda))

#INITIALIZE EMPTY MATRICES FOR STORING PREDICTION AND ERRORS
PRED_IN <- matrix(NA, nrow = dim(training)[1], ncol=length(lambda))
PRED_OUT <- matrix(NA, nrow = dim(holdout)[1], ncol=length(lambda))
E_IN <- matrix(NA, nrow = 1, ncol=length(lambda))
E_OUT <- matrix(NA, nrow = 1, ncol=length(lambda))

for (i in 1:length(lambda)){
  
  #COMPUTE PSEUDOINVERSE SOLUTION
  BETA_RIDGE[,i] <- solve(t(X)%*%X+lambda[i]*diag(dim(t(X)%*%X)[1]))%*%t(X)%*%y
  
  #COMPUTE PREDICTIONS IN AND OUT-OF-SAMPLE
  PRED_IN[,i] <- X%*%BETA_RIDGE[,i]
  PRED_OUT[,i] <- X_holdout%*%BETA_RIDGE[,i]
  
  #COMPUTE PREDICTION ERRORS (MSE) IN AND OUT-OF-SAMPLE
  E_IN[i] <- sqrt(mean((y-PRED_IN[,i])^2))
  E_OUT[i] <- sqrt(mean((y_holdout-PRED_OUT[,i])^2))
  }

#STORE ERRORS VS. LAMBDAS IN SEPARATE DATAFRAMES
df_IN <- data.frame(cbind(Error=as.numeric(E_IN), Lambda=lambda))
df_OUT <- data.frame(cbind(Error=as.numeric(E_OUT), Lambda=lambda))

ggplot(df_IN, aes(y=Error, x=Lambda)) +
  geom_line(color='blue') +
  geom_line(data=df_OUT, color='red') +
  ggtitle("E_IN & E_OUT VS. REGULARIZATION PARAMETER (LAMBDA)") +
  theme(plot.title = element_text(hjust = .5))
  
#REPORT MINIMUM E_OUT ESTIMATE FROM BEST REGULARIZED MODEL
(min(df_OUT$Error))

#RECOVER OPTIMAL LAMBDA
(Opt_Lambda <- df_OUT$Lambda[which.min(df_OUT$Error)])

#REPLOT WITH MINIMUM ERROR IDENTIFIED
ggplot(df_OUT, aes(y=Error, x=Lambda)) +
  geom_line(color='red') +
  geom_vline(xintercept=Opt_Lambda, color='red', lty=2) +
  ggtitle("E_OUT VS. REGULARIZATION PARAMETER (LAMBDA)") +
  theme(plot.title = element_text(hjust = .5))

#################################################
##ALTERNATIVES USING MASS AND LMRIDGE LIBRARIES##
#################################################

library(broom) #FOR tidy() AND glance()
library(MASS) #FOR lm.ridge()
library(lmridge) #FOR lmridge()

##################
##MODEL BUILDING##
##################
unreg_mod<-lm(units_sold~.,df) #BUILD UNREGULARIZE MODEL AS POINT OF COMPARISION
reg_mod<-lm.ridge(units_sold ~ ., df, lambda=seq(0,.5,.01)) #BUILD REGULARIZED MODEL

##DIAGNOSTIC OUTPUT##
summary_reg <- tidy(reg_mod)
head(summary_reg, 10)

reg_mod$coef #PULL MODEL COEFFICIENTS FOR EACH REGULARIZATION VALUE (LAMBDA)
print(reg_mod) #OR ALTERNATIVELY
glance(reg_mod) #USING BROOM PACKAGE TO EXRACT OPTIMAL LAMBDA

summary(unreg_mod) #REPORT DIAGNOSTICS FOR UNREGULARIZED MODEL

#TO SEE HOW BETAS CHANGE WITH LAMBDA
plot(reg_mod) 

#OR ALTERNATIVELY, SEE PARAMETER SHRINKAGE WITH GGPLOT
ggplot(summary_reg, aes(lambda, estimate, color=term)) +
  geom_line()

#PLOTTING MODEL COMPLEXITY (LAMBDA) VS VALIDATION ERROR
ggplot(summary_reg, aes(lambda, GCV)) +
  geom_line() +
  geom_vline(xintercept=glance(reg_mod)$lambdaGCV, col='red', lty=2)

##USING lmridge PACKAGE (DIFFERENT THAN lm.ridge()) TO GENERATE PREDICTIONS (EASIER)
reg1<-lmridge(units_sold~., df, K=.06)
summary(reg1) #THIS LINE RUNS VERY SLOW BECAUSE IT RUNS VALIDATION
coef(reg1)
pred<-predict(reg1, df) #GENERATE PREDICTIONS FROM ALL MODELS FOR ALL LAMBDAS
dim(pred) #DIMENSIONS OF PREDICTION ON ALL MODELS
