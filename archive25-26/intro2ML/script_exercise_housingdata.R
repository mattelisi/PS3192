rm(list=ls())

d_train <- read.csv("california_housing_train.csv")

# -----------------------------------------------------------
# custom function for doing LOO-cross-validation

loocrossval <- function(mydata, formula){
  n <- nrow(mydata)  # Total number of observations
  preds <- numeric(n)  # Placeholder for predictions
  
  # Extract the outcome name from the formula
  outcome_var <- all.vars(as.formula(formula))[1]
  
  # Leave-One-Out Cross-Validation
  for (i in 1:n) {
    
    # Define training and test sets
    train_data <- mydata[-i, ]  # All except the i-th observation
    test_data  <- mydata[i,  ]  # The i-th observation
    
    # Fit the model
    model <- lm(formula, data = train_data)
    
    # Predict for the left-out observation
    preds[i] <- predict(model, newdata = test_data)
  }
  
  # Compute Mean Squared Error
  mse_loo <- mean((mydata[[outcome_var]] - preds)^2)
  rsquared_loo <- 1 - (mse_loo/mean((mydata[[outcome_var]] - mean(mydata[[outcome_var]]))^2))
  
  # return results as named vector
  res <- c(mse_loo, rsquared_loo)
  names(res) <- c("MSE","r-squared")
  
  # result
  return(res)
}

# -----------------------------------------------------------
# create all possible formulas combinations

var_list <- colnames(d_train[,-which(colnames(d_train)=="medv")])
dependent_var <- "medv"

# empty list to hold all formulae
all_formulas <- list()

# generate combinations of variables
for(i in 1:length(var_list)){
  combinations <- combn(var_list, i)
  num_combinations <- ncol(combinations)
  
  # loop over combinations and create formulas
  for(j in 1:num_combinations){
    formula <- paste(dependent_var, "~", 
                     paste(combinations[,j], collapse="+"))
    
    all_formulas <- c(all_formulas, formula)
  }
}

# -----------------------------------------------------------
# fit all models on training set, and evaluate 
# their performance using a cross-validation

m <- list() # an empty list
results <- data.frame()

# warning: this step may take some time to complete!
for (i in 1:length(all_formulas)) {
  m[[i]] <- lm(all_formulas[[i]], data = d_train)
  
  crossval_res <- loocrossval(d_train, all_formulas[[i]])
  
  # print some output to show progress
  cat("model ",i, "out of ", length(all_formulas), "completed:\n")
  print(crossval_res)
  cat("\n\n")
  
  results <- rbind(results, 
                   data.frame(MSE= crossval_res["MSE"],
                            r_squared = crossval_res["r-squared"],
                            formula = all_formulas[[i]]))
}

saveRDS(results, "cali_results.RDS")

# -----------------------------------------------------------
# find best model

best_formula <- results$formula[results$MSE==min(results$MSE)]

# LOO MSE error of best model
results$MSE[results$MSE==min(results$MSE)] #  13.27313

# LOO r-squared of best model
results$r_squared[results$MSE==min(results$MSE)] # ] 0.8486196

# -----------------------------------------------------------
# Now test best model with new data

# estimate the model parameters using training set data
best_model <- lm(best_formula, data = d_train)

# Predict values of test data
d_test <- read.csv("california_housing_test.csv")
preds <- predict(best_model, newdata = d_test)

# Calculate Mean Squared Error
mse <- mean((d_test$medv - preds)^2)
1 - (mse/mean((d_test$medv- mean(d_test$medv))^2))

# -----------------------------------------------------------

results$test_MSE <- NA
results$test_rsquared <- NA

MSE_tot_test <- mean((d_test$medv- mean(d_test$medv))^2)

for(i in 1:nrow(results)){
  
  m_i <- lm(results$formula[i], data = d_train)
  preds_i <- predict(m_i, newdata = d_test)
  
  results$test_MSE[i] <- mean((d_test$medv - preds_i)^2)
  results$test_rsquared[i] <- 1 - (results$test_MSE[i] / MSE_tot_test)
  
}

with(results, plot(MSE, test_MSE, pch=21, cex=0.2, col=rgb(0,0,0,0.3), 
                   xlim=c(10, 90), 
                   ylim=c(10, 90),
                   xlab="MSE train set (cross-validated)",
                   ylab="MSE test set"))

points(results$MSE[results$MSE==min(results$MSE)], mse, pch=19, col="blue")

abline(a=0, b=1, lty=2)



