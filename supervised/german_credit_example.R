library(rpart)
library(rpart.plot)
library(caret)
library(partykit)

rm(list=ls())
hablar::set_wd_to_script_path()


# data
data(GermanCredit)
str(GermanCredit)

# estimate and evaluate an un-pruned tree
credit_tree <- rpart(Class  ~ ., data = GermanCredit,
                     minsplit = 10, 
                     xval=1000)

# the results of the cross-validation is in this table
credit_tree$cptable

printcp(credit_tree)

pruned_tree <- prune(credit_tree, cp=0.01)
printcp(pruned_tree)

cp_data <- as.data.frame(credit_tree$cptable)
colnames(cp_data) <- c("cp", "nsplit", "rel_error", "xerror", "xstd")

# Compute number of terminal nodes
cp_data$size <- cp_data$nsplit + 1
cp_data$xstd <- cp_data$xstd

# The errors are normalised relative to the model with no-splits
# to put them in the natural scale we need to multiply by the fraction of errors in the 
# no split model

# this the same as the error we make always predicting the majority class:
table(GermanCredit$Class)

# we can re-scale the errors as follow
cp_data[,c("rel_error", "xerror", "xstd")] <- cp_data[,c("rel_error", "xerror", "xstd")] * 300/(300 + 700)

# Plot misclassification error vs. number of terminal nodes
ggplot(cp_data, aes(x = size)) +
  geom_line(aes(y = rel_error, color = "Training"), linewidth = 1) +
  geom_line(aes(y = xerror, color = "Cross-validation"), linewidth = 1) +
  geom_errorbar(aes(y = xerror, ymin=xerror-xstd, ymax=xerror+xstd), color="blue", width=0.1, linewidth = 1) +
  #geom_point(aes(y = xerror), size = 3, shape = 21, fill = "white") +
  #geom_hline(aes(yintercept = min(xerror) + xstd[which.min(xerror)]), linetype = "dotted") +
  geom_point(aes(x = size[which.min(xerror)], y = min(xerror)), color = "purple", size = 8, shape = 1) +
  labs(
    x = "Number of terminal nodes",
    y = "Fraction of misclassifications",
  ) +
  scale_color_manual(values = c("Training" = "red", "Cross-validation" = "blue"), name="") +
  theme_minimal()


credit_tree <- rpart(Class  ~ ., data = GermanCredit,
                     cp = 0, 
                     minsplit = 10, 
                     xval=1000)