---
title: "PS3192: Data science for psychology"
subtitle: "<br>"
author: "Matteo Lisi"
format:
  revealjs:
    logo: img/logo-small-london-cmyk.jpg
    footer: "PS3192 25-26"
    incremental: true  
    auto-stretch: false
    code-fold: false   # don’t fold
    code-line-numbers: false
    theme: [default, matteo_rhul.css]
editor: source
keep-md: true
filters: [bg_style.lua]
---


::: {.cell}

:::



# 

##  PS3192 : Real World Data Science

::: nonincremental

> Learning Outcomes:
> 
> 1. Demonstrate proficiency in data science skills by using programming languages
(e.g., R) to import, prepare, and visualize data effectively.
> 
> 2. Apply appropriate statistical learning techniques to solve data science problems,
and accurately interpret and communicate results.
> 
> 3. Demonstrate critical thinking skills regarding the strengths, limitations, and contexts
for various statistical learning methods, including algorithmic bias, transparency,
and ethical implications.

:::

## Module outline

::: {style="font-size: 70%;"}
| week | day | time | lecturer | topic |
|--------------|--------------|--------------|--------------|----------------|
| 18 | 23 Jan | 10:00 - 12:00 | Matteo | Advanced R programming |
| 19 | 30 Jan | 10:00 - 12:00 | Hirotaka | Data Importing, Cleaning, and Processing |
| 20 | 6 Feb | 10:00 - 12:00 | Hirotaka | Data Visualisation |
| 21 | 13 Feb | 10:00 - 12:00 | Matteo | Introduction to Machine Learning Concepts |
| 22 | 20 Feb | 10:00 - 12:00 | Matteo | Classification and Supervised Learning |
| 23 | 27 Feb | 10:00 - 12:00 | Hirotaka | Reproducible Analyses and Reporting with R Markdown |
| 24 | 6 March | 10:00 - 12:00 | Matteo | Clustering and Unsupervised Learning |
| 25 | 13 March | 10:00 - 12:00 | Jonas | Introduction to Matlab |
| 26 | 20 March | 10:00 - 12:00 | Jonas | Mind reading with a classifier |
| 27 | 27 March | 10:00 - 12:00 | Matteo | Large Language Models |
:::

:::: {style="font-size: 75%;"}

All workshops in Bourne PC Lab.

::::

## Lecturers

::: {.column width="33%"}
![](ML.jpeg) <!-- {.nostretch fig-align="center" width="70%"} --> [Matteo Lisi](https://mlisi.xyz/)
:::

::: {.column width="34%"}
![](hiro.png)

[Hirotaka Imada](https://himada2018.github.io/)
:::

::: {.column width="33%"}
![](jonas.png)\

[Jonas Larsson](https://pure.royalholloway.ac.uk/en/persons/jonas-larsson)
:::

## Assessment

-   Coursework contributes 100% to final course grade

-   Create a portfolio by answering any 3 of 5 questions (Word Count: 5,000)

-   One question will be essay-based, the other will include practical & quantitative aspects.

-   Details to follow soon.


#

## What is 'data science' ?

- Not that different from statistics, actually.

- The mathematical models are mostly the same.

- However, compared to statistics, there is a shift in focus, from **explanation** to **prediction**

::: fragment

|                | **Statistics (in science)** | **Data science / ML**     |
| -------------- | --------------------------- | ------------------------- |
| Primary goal   | Explanation                 | Prediction                |
| Main question  | *Why does this happen?*     | *How accurately can we predict it?*  |
| Focus          | Parameters, effects         | Accuracy, generalisation  |
| Typical output | Coefficients, $p$-values    | Predictions, error        |
| Evaluation     | Is the effect "real"?       | Does it work on new data? |

:::

- In practice the demarcation isn't very clear, and there is a lot of overlap.


## Activity: analysis of `earnings` dataset

This dataset contains the annual earnings (in $), the heights (in inches) and the sex (male/female) of a random sample (N=1192) of adult Americans, surveyed in the 1990.




::: {.cell}
::: {.cell-output .cell-output-stdout}

```
'data.frame':	1192 obs. of  3 variables:
 $ female: int  0 1 1 1 1 1 1 0 0 0 ...
 $ earn  : num  50000 60000 30000 50000 51000 9000 29000 32000 2000 27000 ...
 $ height: int  74 66 64 63 63 64 62 73 72 72 ...
```


:::
:::



Open the worksheet at [mlisi.xyz/PS3192/advanced_R/worksheet.html](https://mlisi.xyz/PS3192/advanced_R/worksheet.html). Link also available on Moodle page.


## Optional reading

Chapters 1-3 

![](img/book_cover.png){fig-align="center" width="30%"}


## Optional exercise

Load the `california_housing_train.csv` data from Moodle and find the best predictive model to predict house value (`medv`).

 

::: nonincremental

1.  Aim to build a model that generalizes well to new data.

2.  Use any approach you find suitable—incorporate interactions, higher-order (polynomial) terms, and evaluate performance using hold-out samples.

3.  Once you've selected a model, assess its predictive performance on out-of-sample data.

4.  In our next workshop I will release a test set on Moodle, allowing you to check your model’s true predictive performance on unseen data.

:::


## Data Dictionary for California Housing Dataset

::: {style="font-size: 60%;"}
| Variable | Description |
|-----------------------------------|-------------------------------------|
| medv | Median value of owner-occupied homes (in \$1000s) |
| crim | Per capita crime rate by town |
| zn | Proportion of residential land zoned for large lots |
| indus | Proportion of non-retail business acres per town |
| chas | Charles River dummy variable (1 if tract bounds river, 0 otherwise) |
| nox | Nitrogen oxide concentration (parts per 10 million) |
| rm | Average number of rooms per dwelling |
| age | Proportion of owner-occupied units built before 1940 |
| dis | Weighted distance to employment centers |
| rad | Index of accessibility to highways |
| tax | Property tax rate per \$10,000 |
| ptratio | Pupil-teacher ratio by town |
| lstat | \% lower status of the population |
| lon | longitude |
| lat | latitude |

