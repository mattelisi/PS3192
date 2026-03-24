---
title: "Artificial intelligence"
subtitle: "<br>PS3192: Data science for psychology"
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



## Outline for today

:::: nonincremental
-   A brief history of artificial intelligence
-   Large language models & data-science applications

::::

# A history of AI

## 

 
 

Modern AI is built on **neural networks** — inspired by the brain, but increasingly diverging from it.


##  {background-color="#202A30"}

![_'Brainbow Hippocampus'_ by Greg Dunn.](img/Brainbow-Hippocampus-in-Color-2020-remastered.webp)


## Biological neurons

 
 

::: fragment
![There are about $\approx$ 86 billions neurons in the brain, and perhaphs 100 trillion to 1 quadrillion synapses.](img/Blausen_0657_MultipolarNeuron.png){fig-align="center"   width="60%"}
:::

## Artificial neurons

 
 

McCulloch & Pitts (1943) proposed the first **artificial neuron** model

 

![](img/MCPneuron.png){fig-align="center"   width="60%"}


::: fragment

- A simple unit of computation inspired by the biological neuron
- Takes *binary inputs*, computes a *weighted sum*, fires if it exceeds a threshold
- Can be combined to implement *any Boolean logic*
- *But:* weights are fixed — it cannot learn
:::


## Artificial neurons

 

Artificial neurons in modern network uses also continuous inputs and more sophisticated activation functions

![](img/AN.png){fig-align="center"   width="80%"}

## Learning {.smaller}

:::: {.columns}

::: {.column width="45%"}
**Donald Hebb (1949)**

> *"Neurons that fire together, wire together"*

- Learning as strengthening of synaptic connections between co-active neurons

 

**Frank Rosenblatt (1962)**

- The **Perceptron**: a single layer network that can learn weights from data
- Weights encode knowledge!
:::


::: {.column width="55%"}

![](img/rosenblatt.jpg){fig-align="center"   width="50%"}


![](img/retina-perceptron.png){fig-align="center"   width="60%"}


:::

:::: 

## Limits — The XOR Problem & Minsky–Papert (1969) {.smaller}

:::: {.columns}

::: {.column width="50%"}
**The good news:**

*Perceptron Convergence Theorem* — if a linear solution exists, the perceptron **is guaranteed to find it**.

**The bad news:**

Many real problems are **not linearly separable** — the classic case is XOR:

| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single straight line can separate the 1s from the 0s.

Minsky & Papert (1969) formalised these limitations — **setting off the first AI Winter**.
:::

::: {.column width="50%"}


::: {.cell}
::: {.cell-output-display}
![](AI_slides_files/figure-revealjs/xor-1.png){width=432}
:::
:::


:::

::::



