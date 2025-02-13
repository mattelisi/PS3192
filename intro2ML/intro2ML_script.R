rm(list=ls())
hablar::set_wd_to_script_path()

# intro to ML notes
library(rpart)
library(rpart.plot)
library(tidyverse)
library(palmerpenguins)
library(partykit)

# tree example

p_m <- rpart(species~ ., data = penguins)
plot(p_m)
rpart.plot(p_m)

# Prettier plot
rpart.plot(
  p_m, 
  type = 2,         # Split labels are drawn at the decision nodes
  #extra = 104,      # Show predicted class and probability percentages
  box.palette = "RdYlGn",  # Color nodes by prediction (red → yellow → green)
  fallen.leaves = TRUE,    # Arrange terminal nodes at the same level
  tweak = 1,     # Slightly increase text size for readability
  shadow.col = "gray",  # Add shadows to the boxes
  branch = 0.6    # Make branches less steep
)

p_m2 <- as.party(p_m)
plot(p_m2, digits = 0, id = FALSE, terminal_panel = node_barplot(p_m2, id = FALSE, rot=0),
     inner_panel = node_inner(p_m2, id = FALSE, pval = FALSE),
     gp=gpar(fontsize=8,
             height = 21))
     

# overfitting example

set.seed(23)

# assume a "true" underlying function
x <- seq(-10, 10, length.out = 200)
y <- x + x^2 -0.2*x^3
plot(x,y, type="l", col="blue", lwd=2, ylim=c(-200, 300))

# simulate some noisy observations
x_obs <- x[sample(1:200, 10)]
y_obs <- x_obs + x_obs^2  -0.2*x_obs^3+ rnorm(10, mean=0, sd=50)
points(x_obs, y_obs, pch=19)

# observed data
d <- data.frame(y=y_obs,
                x=x_obs)

# models
# m <- list()
# m[[1]] <- lm(y~1, d)
# m[[2]] <- lm(y~x, d)
# m[[3]] <- lm(y~x +I(x^2), d)
# m[[4]] <- lm(y~x +I(x^2) +I(x^3), d)
# m[[5]] <- lm(y~x +I(x^2) +I(x^3) +I(x^4), d)
# m[[6]] <- lm(y~x +I(x^2) +I(x^3) +I(x^4) +I(x^5), d)
# m[[7]] <- lm(y~x +I(x^2) +I(x^3) +I(x^4) +I(x^5) +I(x^6), d)

m <- list()
m[[1]] <- lm(y~1, d)
for (p in 0:8) {
  formula <- as.formula(paste("y ~", paste0("I(x^", 1:p, ")", collapse="+")))
  m[[p + 1]] <- lm(formula, data = d)
}

# error on trainin set
training_error <- rep(NA, length(m))
for(i in 1:length(m)){
  pred_y <- predict(m[[i]])
  training_error[i] <- sqrt(sum((d$y -  pred_y)^2)/length(pred_y ))
}
# 
# # add to plot
# for(i in 1:length(m)){
#   pred_y <- predict(m[[i]], newdata=data.frame(x))
#   lines(x,y)
# }

# # more concise
# models <- lapply(0:7, function(p) lm(as.formula(paste("y ~", paste0("I(x^", 1:p, ")", collapse = " + "))), data = d))


# r2_values <- sapply(m, function(m) c(R2 = summary(m)$r.squared, Adj_R2 = summary(m)$adj.r.squared))
# 
# # Convert to a data frame for readability
# r2_df <- as.data.frame(t(r2_values))
# names(r2_df) <- c("R^2", "Adjusted R^2")

# test true predictive performance on unseen data
x_new <- runif(500, min=-10, max=10)
y_new <- x_new + x_new^2  -0.2*x_new^3 + rnorm(500, mean=0, sd=50)
d_new <- data.frame(x = x_new, y=y_new)
#points(x_new, y_new, pch="*", col="red")

test_error <- rep(NA, length(m))
for(i in 1:length(m)){
  pred_y <- predict(m[[i]], newdata=d_new)
  test_error[i] <-  sqrt(sum((d_new$y -  pred_y)^2)/length(pred_y ))
}

plot(1:length(m), training_error,
     xlab="n. parameters", ylab="error",
     ylim=c(0,1800), type="o", col="blue")

lines(1:length(m), test_error, type="o", col="red")


####


library(ggplot2)
library(viridis)  # For a good perceptual color scale

# True function
x <- seq(-10, 10, length.out = 200)
y <- x + x^2 - 0.2 * x^3

# Simulated noisy observations
set.seed(23)
x_obs <- x[sample(1:200, 10)]
y_obs <- x_obs + x_obs^2 - 0.2 * x_obs^3 + rnorm(10, mean = 0, sd = 50)
d <- data.frame(x = x_obs, y = y_obs)

# Fit models of increasing complexity
m <- list()
m[[1]] <- lm(y ~ 1, d)  # Intercept-only model
for (p in 1:8) {
  formula <- as.formula(paste("y ~", paste0("I(x^", 1:p, ")", collapse = "+")))
  m[[p + 1]] <- lm(formula, data = d)
}

# Generate predictions for plotting
preds <- data.frame(x = rep(x, 9), Degree = rep(0:8, each = length(x)))
preds$y_hat <- unlist(lapply(m, function(mod) predict(mod, newdata = data.frame(x = x))))

# Define a perceptually uniform color palette
palette_colors <- viridis(9)

# Create the ggplot
ggplot() +
  # True function
  geom_line(aes(x, y), color = "black", lwd = 2) +
  # Fitted models
  geom_line(data = preds, aes(x = x, y = y_hat, color = factor(Degree)), lwd = 1) +
  # Noisy observations
  geom_point(aes(x_obs, y_obs), color = "black", size = 4) +
  # Aesthetic adjustments
  scale_color_manual(values = palette_colors, name = "Polynomial Degree") +
  labs(x = "x", y = "y", title = "True Function and Polynomial Fits") +
  theme_minimal()+
  coord_cartesian(ylim=c(-200,300))


# exercise

# Load necessary library
if (!requireNamespace("mlbench", quietly = TRUE)) install.packages("mlbench")
library(mlbench)

# Load the California Housing dataset
data("BostonHousing2", package = "mlbench")

# Remove non-numeric columns (town, tract, cmedv) for a clean regression dataset
bh_data <- subset(BostonHousing2, select = -c(town, tract, cmedv))

# Set seed for reproducibility
set.seed(123)

# Create a random index for splitting
n <- nrow(bh_data)
train_index <- sample(seq_len(n), size = n/2, replace = FALSE)

# Split data into 50% training and 50% test
train_data <- bh_data[train_index, ]
test_data  <- bh_data[-train_index, ]

# Save as CSV
write.csv(train_data, "california_housing_train.csv", row.names = FALSE)
write.csv(test_data, "california_housing_test.csv", row.names = FALSE)

# Print confirmation
print("Train and test datasets saved as CSV.")

