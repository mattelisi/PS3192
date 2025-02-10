# dataset supervised


# --------------------------------------------------------------

# 1. Caravan Insurance Data (from the ISLR or ISLR2 package)
# Key Points
# 
# Task: Predict whether a household purchases a caravan insurance policy.
# Purchase is a binary factor: "Yes" or "No".
# Contains over 80 features describing demographic and socio-economic variables.
# Often used to illustrate classification techniques in the “Introduction to Statistical Learning” book.

# install.packages("ISLR2")  # if not already installed
library(ISLR2)

data("Caravan")
str(Caravan)

# The target variable is Caravan$Purchase
table(Caravan$Purchase)

# You can model it with, for example, rpart or logistic regression.


# -----------------------------------------------------------------
# 2. German Credit Data (from the caret or mlbench package)
# Key Points
# 
# Task: Classify customers as “Good” or “Bad” credit risks based on demographic and financial features.
# This is a classic dataset from the UCI Machine Learning Repository.
# Contains both numerical and categorical predictors (loan amount, duration, age, etc.).

# install.packages("caret")
library(caret)

data("GermanCredit")  # Provided by the caret package
str(GermanCredit)

# The target variable can be GermanCredit$Class (Good/Bad).


# -----------------------------------------------------------------
# 3. insuranceData Package (multiple datasets)
# There is a dedicated package on CRAN called insuranceData that includes several datasets related to personal or property insurance (auto, health, etc.). Some are more geared toward regression (claim amounts), but you can often transform them into classification tasks (e.g., “Has claim / No claim”).
# 
# dataCar, dataHealth, dataIncome, etc.
# You can explore each dataset to see if there’s a straightforward binary outcome or if you might need to derive one.

# install.packages("insuranceData")
library(insuranceData)

data("dataCar")
?dataCar





