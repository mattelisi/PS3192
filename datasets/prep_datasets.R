rm(list=ls())
hablar::set_wd_to_script_path()
library(tidyverse)

library(fivethirtyeight)
data(college_recent_grads)
write.csv(college_recent_grads, "college_recent_grads.csv",row.names=F)



# Wells
d <- read.table("wells.dat",header=T)
write.csv(d, "wells.csv",row.names=F)

# #
# setwd("/mnt/sda2/matteoHDD/git_local_HDD/PS212-regression/matteo_examples")

# titanic
d <- read.csv("titanic.csv")
str(d)
library(rpart)
library(rpart.plot)
titanic_model <- rpart(survived ~ sex + age + passengerClass, data = d)
summary(titanic_model)
plot(titanic_model)
rpart.plot(titanic_model, main = "Pruned Decision Tree")
d$predicted <- round(predict(titanic_model))
confmat <- confusionMatrix(data=factor(d$predicted), reference = factor(d$survived))
confmat

# heatdisease
library(MLDataR)
data(heartdisease)
str(heartdisease)
hd_m <- rpart(HeartDisease ~ ., data = heartdisease)
rpart.plot(hd_m)
write.csv(heartdisease, "heartdisease.csv",row.names=F)


# Iris
data(iris)
str(iris)
m <- rpart(Species  ~ ., data = iris)
rpart.plot(m)
write.csv(iris, "iris.csv",row.names=F)

# 
data(oil)
str(fattyAcids)
str(oilType)
d <- fattyAcids
d$oilType <- oilType
m <- rpart(oilType  ~ ., data = d)
rpart.plot(m)
summary(m)
write.csv(d, "oiltype.csv",row.names=F)
d <- read.csv("oiltype.csv")
str(d)

# Make predictions (use type = "class" to get class labels)
predictions <- predict(m, d, type = "class")

# Compute the confusion matrix
conf_matrix <- confusionMatrix(predictions, d$oilType)
print(conf_matrix)

# # Split data into training (70%) and test (30%) sets
# trainIndex <- createDataPartition(d$oilType, p = 0.7, list = FALSE)
# trainData <- d[trainIndex, ]
# testData <- d[-trainIndex, ]
# 
# # Train the model
# m <- rpart(oilType ~ ., data = trainData)
# 
# # Make predictions on test data
# testPredictions <- predict(m, testData, type = "class")
# 
# # Compute confusion matrix
# conf_matrix_test <- confusionMatrix(testPredictions, testData$oilType)
# print(conf_matrix_test)

# forest mapping
# https://archive.ics.uci.edu/dataset/333/forest+type+mapping
# This data set contains training and testing data from a remote sensing study which mapped different forest types based on their spectral characteristics at visible-to-near infrared wavelengths, using ASTER satellite imagery. The output (forest type map) can be used to identify and/or quantify the ecosystem services (e.g. carbon storage, erosion protection) provided by the forest.
d <- rbind(read_csv("training.csv"), read_csv("testing.csv"))
str(d)
m <- rpart(class ~ ., data = d)
rpart.plot(m)
summary(m)
predictions <- predict(m, d, type = "class")
conf_matrix <- confusionMatrix(predictions, factor(d$class))
print(conf_matrix)
write.csv(d,"forest_mapping.csv", row.names=F)

# autism
library(foreign)
d <- read.arff("./autism/autistic+spectrum+disorder+screening+data+for+children/Autism-Child-Data.arff")
str(d)
# ataset related to autism screening of adults that contained 20 features to be utilised for further analysis especially in determining influential autistic traits and improving the classification of ASD cases. In this dataset, we record ten behavioural features (AQ-10-Child) plus ten individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science. 
d2 <- d %>%
  rename(sex = gender,
         ASD_diagnosis) %>%
  select(-age_desc, -result, -relation, -used_app_before)

str(d2)
m <- rpart(ASD_diagnosis ~ ., data = d2)
rpart.plot(m)
predictions <- predict(m, d2, type = "class")
conf_matrix <- confusionMatrix(predictions, factor(d2$ASD_diagnosis))
print(conf_matrix)
write.csv(d2, "autism.csv",row.names=F)
str(d2)

d2 <- read.csv("autism.csv")
str(d2)

#########################################################################################
# mixture models

# faithful
write.csv(faithful, "faithful.csv",row.names=F)
library(mclust)

gmm_model <- Mclust(faithful)

# Summary of the model
summary(gmm_model)

# Plot the BIC values for model selection
plot(gmm_model, what = "BIC")

# Plot the classification results
plot(gmm_model, what = "classification")

# Scatter plot with cluster assignments
plot(faithful, col = gmm_model$classification, pch = 19,
     xlab = "Eruption Time (min)", ylab = "Waiting Time (min)",
     main = "Gaussian Mixture Model Clustering (faithful dataset)")
legend("topright", legend = paste("Cluster", unique(gmm_model$classification)), 
       col = unique(gmm_model$classification), pch = 19)

# wine
library(HDclassif)
data(wine)
str(wine)

# 1) Alcohol
# 2) Malic acid
# 3) Ash
# 4) Alcalinity of ash  
# 5) Magnesium
# 6) Total phenols
# 7) Flavanoids
# 8) Nonflavanoid phenols
# 9) Proanthocyanins
# 10)Color intensity
# 11)Hue
# 12)OD280/OD315 of diluted wines
# 13)Proline  
colnames(wine) <- c("class","alcohol","malic_acid","ash","ash_alcalinity","magnesium","tot_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","X","proline")
wine_data <- wine[, -1] %>%
  dplyr::select("alcohol","malic_acid","magnesium","tot_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity","hue","proline")

gmm_model <- Mclust(wine_data, G=1:5)
summary(gmm_model)
plot(gmm_model, what = "BIC")
plot(gmm_model, what = "classification")
write.csv(wine_data, "wine.csv",row.names=F)

# 
# library(mlbench)
d <- read_delim("data_banknote_authentication.txt")
colnames(d) <- c("variance","skewness","kurtosis","entropy","class")
str(d)
write.csv(d, "banknote_authentication.csv",row.names=F)
d <- read.csv("banknote_authentication.csv")

m <- rpart(factor(class) ~ ., data = d)
rpart.plot(m)
predictions <- predict(m, d, type = "class")
confusionMatrix(predictions, factor(d$class))

gmm_model <- Mclust(d[,-5], G=1:9)

# Summary of the model
summary(gmm_model)

# Plot the BIC values for model selection
plot(gmm_model, what = "BIC")

# Plot the classification results
plot(gmm_model, what = "classification")


# ##
# # install.packages("psych")  
# library(psych)
# 
# # Load a sample Big Five dataset
# data(bfi)  # Part of the psych package
# str(bfi)
# bfi_data <- bfi[, 1:5]  # 


# fixations
d <- rbind(read_delim("/mnt/sda2/matteoHDD/git_local_HDD/PS3192/datasets/gazedata_afraid_2_M_J_.txt"),
           read_delim("/mnt/sda2/matteoHDD/git_local_HDD/PS3192/datasets/gazedata_afraid_2_M_S_.txt"))
str(d)
gazed <- d %>%
  filter(x>img_left & x < img_right,
         y>img_top & y < img_bottom) %>%
  mutate(x = (x-img_left)/(img_right-img_left),
         y = 1 - (y - img_top)/(img_bottom-img_top)) %>%
  dplyr::select(x,y)

gmm_model <- Mclust(gazed, G=1:9)
summary(gmm_model)
plot(gmm_model, what = "BIC")
plot(gmm_model, what = "classification")

write.csv(gazed, "gazedata.csv",row.names=F)
