library(rpart)
library(rpart.plot)

# Sample data: classification on the iris dataset
data(iris)

# Grow a relatively large tree by setting a low cp:
tree_model <- rpart(Species ~ ., data = iris,
                    method = "class",
                    control = rpart.control(cp = 0.0001, xval = 10))

# Inspect cross-validation results
printcp(tree_model)
plotcp(tree_model)

# Choose cp value based on the lowest cross-validated error
best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
best_cp

# Prune the tree using the chosen CP
pruned_tree <- prune(tree_model, cp = best_cp)

# Visualize
rpart.plot(pruned_tree, main = "Pruned Decision Tree")
