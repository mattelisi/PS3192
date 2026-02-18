# Packages
library(palmerpenguins)
library(dplyr)
library(xgboost)
library(caret)

#--------------------------------------------------
# 1. Prepare data
#--------------------------------------------------

# Remove missing values
penguins_clean <- penguins |> 
  drop_na()

# Convert species to numeric labels (required by xgboost)
penguins_clean$species <- as.numeric(penguins_clean$species) - 1
# (xgboost requires labels: 0, 1, 2)

# Create model matrix (one-hot encode categorical predictors)
X <- model.matrix(species ~ . -1, data = penguins_clean)
y <- penguins_clean$species

#--------------------------------------------------
# 2. Train-test split
#--------------------------------------------------

set.seed(123)

train_index <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]

y_train <- y[train_index]
y_test  <- y[-train_index]

# Convert to xgboost DMatrix format
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

#--------------------------------------------------
# 3. Set parameters
#--------------------------------------------------

params <- list(
  objective = "multi:softprob",   # multiclass classification
  num_class = 3,
  eval_metric = "mlogloss",
  eta = 0.1,           # learning rate
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8
)

#--------------------------------------------------
# 4. Train model
#--------------------------------------------------

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1
)

#--------------------------------------------------
# 5. Predictions
#--------------------------------------------------

pred_probs <- predict(xgb_model, dtest)

# Convert probabilities to class predictions
pred_matrix <- matrix(pred_probs, ncol = 3, byrow = TRUE)
pred_class <- max.col(pred_matrix) - 1

#--------------------------------------------------
# 6. Evaluate accuracy
#--------------------------------------------------

confusionMatrix(
  factor(pred_class),
  factor(y_test)
)
