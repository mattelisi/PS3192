library(caret)
library(glmnet)  # caret uses this under the hood for method = "glmnet"

d_train <- read.csv("california_housing_train.csv")

d_test <- read.csv("california_housing_test.csv")

set.seed(123)  # for reproducibility

ctrl <- trainControl(
  method = "cv",
  number = 5    # 5-fold CV
)

# include this formula and mention that we set the alpha parameter to use L1 or L2 penalty (lasso vs ridge)
# $$\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat y_i \right)^2 \;+\; \lambda \left[ \alpha \sum |\boldsymbol{\beta}| +(1-\alpha) \sum \boldsymbol{\beta}^2 \right]$$


# Example with your housing data
# medv as outcome, some predictors as example
lasso_grid <- expand.grid(
  alpha  = 1,                     # 1 = lasso
  lambda = 10^seq(-3, 1, length = 20)  # from 0.001 to 10
)

set.seed(123)
lasso_fit <- train(
  medv ~ .,
  data      = d_train,
  method    = "glmnet",
  trControl = ctrl,
  tuneGrid  = lasso_grid
)

# all tuning parameters tested
lasso_fit

# Best tuning parameters
lasso_fit$bestTune

# coefficeint of the best model
coef(lasso_fit$finalModel, lasso_fit$bestTune$lambda)

# note that we are using a different method for the coeff function!

# see all the methods
methods("coef")

# see the specific method. require a second parameter s that is the best penalty value
?coef.glmnet

# Plot performance vs lambda (log scale)
plot(lasso_fit)

# predict on the test test
pred_lasso <- predict(lasso_fit, newdata = d_test)



#################### now to the same for ridge regression

ridge_grid <- expand.grid(
  alpha  = 0,                     # 0 = ridge
  lambda = 10^seq(-3, 1, length = 20)
)

set.seed(123)
ridge_fit <- train(
  medv ~ crim + rm + lstat + ptratio + nox,
  data      = d_train,
  method    = "glmnet",
  trControl = ctrl,
  tuneGrid  = ridge_grid
)

ridge_fit$bestTune

coef(ridge_fit$finalModel, ridge_fit$bestTune$lambda)

plot(ridge_fit)

pred_ridge <- predict(ridge_fit, newdata = d_test)

elastic_grid <- expand.grid(
  alpha  = seq(0, 1, by = 0.25),
  lambda = 10^seq(-3, 1, length = 20)
)


