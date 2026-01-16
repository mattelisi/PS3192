rm(list=ls())
# unsupervised script
hablar::set_wd_to_script_path()

library(mclust)
help(package="mclust") 

#######################################################################################
# plot 1

# Load packages
library(mvtnorm)
library(ellipse)
library(plot3D)

# Define parameters for 3 Gaussians
set.seed(123)
mu <- list(c(0.3,0.5), c(0.5,0.5), c(0.7,0.7))
sigma <- list(matrix(c(0.005,0.005,0.005,0.01),2),
              matrix(c(0.005,-0.006,-0.006,0.01),2),
              matrix(c(0.005,0,0,0.01),2))
weights <- c(0.33, 0.33, 0.33)

# Create grid for plotting
x <- seq(0.1, 0.9, length.out=100)
y <- seq(0.1, 0.9, length.out=100)
grid <- expand.grid(x=x, y=y)

# Mixture density calculation
dmix <- function(x, y, mu, sigma, weights){
  d <- 0
  for(i in 1:length(weights)){
    d <- d + weights[i]*dmvnorm(cbind(x,y), mean=mu[[i]], sigma=sigma[[i]])
  }
  return(d)
}

z <- matrix(dmix(grid$x, grid$y, mu, sigma, weights), 100,100)

# Setup multi-panel plot
par(mfrow=c(1,2))

# Plot 2D confidence ellipses
plot(NA, xlim=c(0.1,0.9), ylim=c(0.1,0.9), xlab="", ylab="", asp=1, main="(a)")
colors <- c("red", "green", "blue")
for(i in 1:3){
  for(level in seq(0.1,0.9, by=0.2)){
    lines(ellipse(sigma[[i]], centre=mu[[i]], level=level), col=colors[i], lwd=2)
  }
}

# Plot 3D density landscape
persp3D(x, y, z, theta=10, phi=25, expand=0.5, col="dark grey", border=NA,
        lighting=TRUE, ltheta=45, shade=0.5, ticktype="detailed", main="(b)", xlab="", ylab="", zlab="", axes=FALSE, bty="n")


#######################################################################################
# animate plot 1
library(animation)

# Define parameters for 3 Gaussians
set.seed(123)
mu <- list(c(0.3,0.5), c(0.5,0.5), c(0.7,0.7))
sigma <- list(matrix(c(0.005,0.005,0.005,0.01),2),
              matrix(c(0.005,-0.006,-0.006,0.01),2),
              matrix(c(0.005,0,0,0.01),2))
weights <- c(0.33, 0.33, 0.33)

# Create grid for plotting
x <- seq(0.1, 1, length.out=100)
y <- seq(0.1, 1, length.out=100)
grid <- expand.grid(x=x, y=y)

# Mixture density calculation
dmix <- function(x, y, mu, sigma, weights){
  d <- 0
  for(i in 1:length(weights)){
    d <- d + weights[i]*dmvnorm(cbind(x,y), mean=mu[[i]], sigma=sigma[[i]])
  }
  return(d)
}

z <- matrix(dmix(grid$x, grid$y, mu, sigma, weights), 100,100)

# Setup multi-panel plot

# theta_seq <-  seq(10, 370, length.out=60)
theta_seq <- c(seq(20, 65, length.out=25), seq(64, 21, length.out=25))

# Setup animation
saveGIF({
  for (theta in theta_seq ) {
    
    par(mfrow=c(1,2),mar=c(5,4,1.5,1)+0.1)
    
    # Plot 2D confidence ellipses
    plot(NA, xlim=c(0.1,1), ylim=c(0.1,1), xlab="", ylab="", asp=1)
    colors <- c("red", "green", "blue")
    for(i in 1:3){
      for(level in seq(0.1,0.9, by=0.2)){
        lines(ellipse(sigma[[i]], centre=mu[[i]], level=level), col=colors[i], lwd=2)
      }
    }
    
    # Plot 3D density landscape
    persp3D(x, y, z, theta=theta, phi=25, expand=0.5, col="brown", border=NA,
            lighting=TRUE, ltheta=theta+35, shade=0.5, ticktype="detailed", xlab="", ylab="", zlab="", axes=FALSE, bty="n")
    
    # persp3D(x, y, z, theta=theta, phi=25, expand=0.6, col="brown", border="white",
    #         lighting=TRUE, ltheta=45, shade=0.5, ticktype="detailed", 
    #         main="Rotating Gaussian Mixture", xlab="", ylab="", zlab="")
  }
}, interval = 0.1, movie.name = "Gaussian_Mixture.gif", ani.width = 1200, ani.height = 600)


#######################################################################################

set.seed(123)
n_obs <- 200
# Known covariance matrices
cov_matrices <- list(
  "rho==-0.7" = matrix(c(1, -0.7, -0.7, 1), 2),
  "rho==0" = matrix(c(1, 0, 0, 1), 2),
  "rho==0.5" = matrix(c(1, 0.5, 0.5, 1), 2),
  "rho==0.85" = matrix(c(1, 0.85, 0.85, 1), 2)
)

# Generate data
rn5 <- MASS::mvrnorm(n_obs, c(0, 0), Sigma = cov_matrices[["rho==-0.7"]])
r00 <- MASS::mvrnorm(n_obs, c(0, 0), Sigma = cov_matrices[["rho==0"]])
r03 <- MASS::mvrnorm(n_obs, c(0, 0), Sigma = cov_matrices[["rho==0.5"]])
r07 <- MASS::mvrnorm(n_obs, c(0, 0), Sigma = cov_matrices[["rho==0.85"]])

# Combine datasets
dat <- data.frame(rbind(rn5, r00, r03, r07))
colnames(dat) <- c("X1", "X2")
dat$correlation <- factor(c(rep("rho==-0.7", n_obs),
                            rep("rho==0", n_obs),
                            rep("rho==0.5", n_obs),
                            rep("rho==0.85", n_obs)),
                          levels = c("rho==-0.7", "rho==0", "rho==0.5", "rho==0.85"))

# Create theoretical ellipses
theoretical_ellipses <- do.call(rbind, lapply(names(cov_matrices), function(rho){
  ell <- ellipse(cov_matrices[[rho]], level=0.95, npoints=100)
  data.frame(X1=ell[,1]*24.741, X2=ell[,2]*5.922, correlation=rho)
}))

# Plot with theoretical ellipses and custom theme
ggplot(dat, aes(x = X1 * 24.741, y = X2 * 5.922)) +
  geom_point(alpha = 0.8, color="white") +
  geom_path(data=theoretical_ellipses, aes(x=X1, y=X2), color="white", size=1.2) +
  facet_grid(. ~ correlation, labeller = label_parsed) +
  labs(x = expression(mu[0]), y = expression(mu[1])) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill="#202A30", color="#202A30"),
    plot.background = element_rect(fill="#202A30", color="#202A30"),
    strip.background = element_rect(fill="#202A30", color="#202A30"),
    strip.text = element_text(color="white"),
    axis.text = element_text(color="white"),
    axis.title = element_text(color="white"),
    panel.grid = element_blank()
  )



#######################################################################################
# create similar mouse data as in the wiki example


set.seed(42)
n_obs <- 300

# Define three Gaussian clusters (mouse shape)
ear1 <- MASS::mvrnorm(n = n_obs/2, mu = c(3, 8), Sigma = matrix(c(0.3, 0, 0, 0.3), 2))
ear2 <- MASS::mvrnorm(n_obs/2, c(8, 8), Sigma = matrix(c(0.3, 0, 0, 0.3), 2))
head <- MASS::mvrnorm(n_obs*2, c(5.5, 5), Sigma = matrix(c(2.5, 0, 0, 2.5), 2))

# Combine into one dataset
dat <- data.frame(rbind(ear1, ear2, head))
colnames(dat) <- c("X1", "X2")

# dat <- dat[sample(nrow(dat)),]
# write_csv(dat,"mouse.csv")

dat$cluster <- factor(c(rep("Ear1", n_obs/2), rep("Ear2", n_obs/2), rep("Head", n_obs*2)))

# Plot dataset (similar to Wikipedia "mouse" example)
ggplot(dat, aes(x = X1, y = X2, color=cluster)) +
  geom_point(alpha=0.8) +
  stat_ellipse(level=0.95, size=1.2) +
  theme_minimal() +
  labs(title = "Mouse-like Gaussian clusters", x="X-axis", y="Y-axis")


#######################################################################################
library(mclust)

dat <- read.csv("mouse.csv")
head(dat)

BIC <- mclustBIC(dat)
plot(BIC)




# Fit the Gaussian mixture model
fit <- Mclust(dat)
summary(fit)

# Add predicted cluster to the dataset
dat$cluster_pred <- factor(fit$classification)

ggplot(dat, aes(x = X1, y = X2, color = cluster_pred)) +
  geom_point(alpha = 0.8) +
  theme_minimal() +
  labs(title = "Gaussian Mixture Model Clustering",
       x = "X-axis", y = "Y-axis",
       color = "GMM cluster")




# Perform k-means clustering (3 clusters)
kmeans_fit <- kmeans(dat[, c("X1", "X2")], centers = 3)

# Add predicted cluster to the dataset
dat$cluster_kmeans <- factor(kmeans_fit$cluster)

# Plot k-means results
ggplot(dat, aes(x = X1, y = X2, color = cluster_kmeans)) +
  geom_point(alpha = 0.8) +
  theme_minimal() +
  labs(title = "K-means Clustering",
       x = "X-axis", y = "Y-axis",
       color = "K-means cluster")




############# plot model selection
fit <- Mclust(dat)

BIC_values <- fit$bic  

# Plot BIC values (same as if you had used mclustBIC(dat))
plot(fit, what = "BIC")



fit <- Mclust(dat)

# Extract BIC values from the fitted model
BIC_values <- fit$bic  

# Plot BIC values (same as if you had used mclustBIC(dat))
plot(fit, what = "BIC")


##################################################### More than 2D

# Load the dataset
data(iris)
df <- iris[, -5]  # Remove species column (we assume it's unknown)

# Fit GMM
fit <- Mclust(df)
summary(fit)

# Extract cluster assignments
iris$cluster <- as.factor(fit$classification)

plot(fit, what = "classification")

library(palmerpenguins)
str(penguins)

dat <- na.omit(penguins[, c("bill_length_mm", "flipper_length_mm", "body_mass_g")])
fit <- Mclust(dat)
summary(fit)

plot(fit, what = "classification")

dat2 <- na.omit(penguins[, c("bill_length_mm", "flipper_length_mm", "body_mass_g", "species")])
clPairs(dat, classification=dat2$species)

