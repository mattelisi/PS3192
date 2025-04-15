rm(list=ls())

library(tidyverse)
library(jpeg)
hablar::set_wd_to_script_path()

# load info
d <- read_csv("datasets/gazedata.csv")
img <- readJPEG("afraid.jpg")

# make plot
ggplot(d, aes(x = x, y = y)) +
  annotation_raster(img, xmin = 0, xmax = 1, ymin = 0, ymax = 1) +
  geom_point(pch=21) 
