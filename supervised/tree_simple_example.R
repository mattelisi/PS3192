library(rpart)
library(rpart.plot)

hablar::set_wd_to_script_path()

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

#-----------------------------------------------------------------
# Penguins

library(rpart)
library(palmerpenguins)
library(partykit)

# tree example

p_m <- rpart(species~ ., data = penguins)
p_m2 <- as.party(p_m)

plot(p_m2, digits = 0, id = FALSE, terminal_panel = node_barplot(p_m2, id = FALSE, rot=0),
     inner_panel = node_inner(p_m2, id = FALSE, pval = FALSE),
     gp=gpar(fontsize=8,
             height = 21))

#######################################

library(rpart)
library(palmerpenguins)
library(ggplot2)

# Remove missing values
penguins <- na.omit(penguins)

# Fit decision tree with deeper splits
p_m <- rpart(
  species ~ ., 
  data = penguins, 
  method = "class", 
  cp = 0,        # No automatic pruning
  minsplit = 2,  # Allow small splits
  maxdepth = 10  # Allow deeper trees
)

# Extract complexity parameter (CP) table
cp_data <- as.data.frame(p_m$cptable)
colnames(cp_data) <- c("cp", "nsplit", "rel_error", "xerror", "xstd")

# Compute number of terminal nodes
cp_data$size <- cp_data$nsplit + 1

# Plot misclassification error vs. number of terminal nodes
ggplot(cp_data, aes(x = size)) +
  geom_line(aes(y = rel_error, color = "Training error"), linewidth = 1) +
  geom_line(aes(y = xerror, color = "Cross-validation error"), linewidth = 1) +
  geom_point(aes(y = xerror), size = 3, shape = 21, fill = "white") +
  geom_hline(aes(yintercept = min(xerror) + xstd[which.min(xerror)]), linetype = "dotted") +
  geom_point(aes(x = size[which.min(xerror)], y = min(xerror)), color = "purple", size = 4, shape = 1) +
  labs(
    x = "Number of terminal nodes",
    y = "Misclassification error",
    title = "Complexity vs. Misclassification Error"
  ) +
  scale_color_manual(values = c("Training error" = "red", "Cross-validation error" = "blue")) +
  theme_minimal()
