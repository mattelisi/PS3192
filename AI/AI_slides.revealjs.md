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

::: nonincremental
-   A brief history of artificial intelligence
-   Large language models & data-science applications
:::

# A history of AI

## 

   

Modern AI is built on **neural networks** — inspired by the brain, but increasingly diverging from it.

##  {background-color="#202A30"}

![*'Brainbow Hippocampus'* by Greg Dunn.](img/Brainbow-Hippocampus-in-Color-2020-remastered.webp)

## Biological neurons

   

::: fragment
![There are about $\approx$ 86 billions neurons in the brain, and perhaphs 100 trillion to 1 quadrillion synapses.](img/Blausen_0657_MultipolarNeuron.png){fig-align="center" width="60%"}
:::

## Artificial neurons

   

McCulloch & Pitts (1943) proposed the first **artificial neuron** model

 

![](img/MCPneuron.png){fig-align="center" width="60%"}

::: fragment
-   A simple unit of computation inspired by the biological neuron
-   Takes *binary inputs*, computes a *weighted sum*, fires if it exceeds a threshold
-   Can be combined to implement *any Boolean logic*
-   *But:* weights are fixed — it cannot learn
:::

## Artificial neurons

 

Artificial neurons in modern network uses also continuous inputs and more sophisticated activation functions

![](img/AN.png){fig-align="center" width="80%"}

## Learning

:::::: columns
::: {.column width="40%"}

::: {style="font-size: 70%;"}
-   **Donald Hebb (1949)** *"Neurons that fire together, wire together"*
:::

<!-- Learning as strengthening of synaptic connections between co-active neurons -->

 

::: {style="font-size: 70%;"}
-   **Frank Rosenblatt (1962)** build the **Perceptron**: a single layer network that can learn weights from data
:::

:::

:::: {.column width="60%"}
::: fragment
![](img/retina-perceptron.png){fig-align="center" width="60%"}

![](img/rosenblatt.jpg){fig-align="center" width="30%"}
:::
::::
::::::

## Limitations

<!-- {.smaller} -->

<!-- #### The XOR problem & Minsky–Papert (1969) -->

 

:::::: columns
:::: {.column width="40%"}
::: {style="font-size: 70%;"}

::::::: {.fragment fragment-index=1}

*Perceptron Convergence Theorem* — if a *linear* solution exists, the perceptron is guaranteed to find it.

:::::::

 

::::::: {.fragment fragment-index=2}

The bad news: many real problems are **not linearly separable** — the classic case is XOR:

| x₁  | x₂  | XOR |
|-----|-----|-----|
| 0   | 0   | 0   |
| 0   | 1   | 1   |
| 1   | 0   | 1   |
| 1   | 1   | 0   |

No single straight line can separate the 1s from the 0s.

:::::::

 

::::::: {.fragment fragment-index=3}

[Minsky & Papert (1969) formalised these limitations — setting off the first **AI Winter**.]{style="color:#198754"}

:::::::

:::
::::

::: {.column width="30%"}

::::::: {.fragment fragment-index=2}



::: {.cell}
::: {.cell-output-display}
![](AI_slides_files/figure-revealjs/xor-1.png){width=336}
:::
:::



:::::::

:::

::: {.column width="30%"}


::::::: {.fragment fragment-index=3}

![](img/perceptron_book_cover.jpg){fig-align="center" width="100%"}

:::::::

:::


::::::

## Single and multi-layer perceptrons


:::::: columns

::: {.column width="50%"}

![](img/single.png){fig-align="center" width="80%"}

:::

::: {.column width="50%"}

:::: fragment

![](img/multi.png){fig-align="center" width="80%"}

::::

:::

::::::

## The first "AI winter" (1970-1980) 

::: nonincremental

::::::: {.fragment fragment-index=1}
- Period of severely reduced funding and interest in neural network research, due to unmet expectations.
:::::::

::::::: {.fragment fragment-index=2}
- Funding and energy moved toward **symbolic AI**, focused on using high-level representations and logic to model human intelligence.
:::::::

::::::: {.fragment fragment-index=3}
- Minsky and Papert, among others, were proponents of symbolic AI approaches and argued that neural networks would fail to scale up beyond toy problems.
:::::::

:::

 

::::::: {.fragment fragment-index=4}
A second AI winter in the late 80s, when it became clear that many buyers and investors expected far more adaptability and robustness than what symbolic AI systems could provide.
:::::::

 

::::::: {.fragment fragment-index=5}
[**_Typical_ AI winter pattern: not "no progress", but expectations rising much faster than capabilities.**]{style="color:#198754"}
:::::::


## Key milestones in neural network AI {.smaller .scrollable background-color="#202A30"}



::: {.cell layout-align="center"}
::: {.cell-output-display}
![](AI_slides_files/figure-revealjs/timeline-1.png){fig-align='center' width=1344}
:::
:::




-  1982: **Backpropagation algorithm**: scalable training of multilayer perceptrons revives interest in neural networks

- 1989 **Convolutional neural network (CNN)**: spatial convolutions inspired by the hierarchical organisation of the visual system

- 1997 **Long short-term memory (LSTM)**: recurrent network that can learn long-range dependencies in sequences, inspired by recurrence in brain circuits

- 2012 **Deep learning**: AlexNet wins the ImageNet competition by a large margin, sparking major industry interest

- 2013 **word2vec** dense distributed word embeddings learned from huge text corpora, allowing to represent semantics with list of numbers (not unlike the brain)

- 2017 _"Attention is all you need"_ paper introduces the **transformer** architecture, which underlies modern large language models (GPT, Claude, Gemini, etc.)


## AlexNet & the ImageNet competition

 

:::::: columns

:::: {.column width="70%"}

![Krizhevsky, Sutskever, & Hinton (2012)](img/Krizhevsky2012_architecture.png){fig-align="center" width="100%"}


::::

:::: {.column width="30%"}

![ImageNet contains over 14 million images categorized into 1000 classes.](img/imagenet.webp){fig-align="center" width="100%"}

::::

::::::

 

::: fragment

The success of AlexNet was due also to recent developments in availability of large scale labelled datasets and general purpose GPU computing.

:::


## How brain-like are neural networks? {.scrollable}

 

Many influential ideas in neural networks were borrowed from biology:

- The artificial neuron as a basic computational unit in a network

- Early learning algorithms (e.g. the perceptron) were inspired by the strengthening of connections between co-active neurons

- Information is stored in the strength of synaptic connections, a central idea in neuroscience

- Convolutional networks were heavily inspired by biological vision (local receptive fields, hierarchical feature detectors)

- Early approaches to language and sequence learning relied on _recurrence_, echoing recurrent connectivity in the brain

- Semantic concepts can be represented in a distributed way: both artificial and biological networks encode information as patterns of activity across many units

 
 

::: fragment

Over time, neural networks increasingly diverged from biology, replacing biological realism with engineering abstractions.

:::


::: fragment

A first example is [backpropagation]{.underline}: it uses the chain rule to propagate errors backward and update connection weights during learning. By contrast, biological neurons are thought to learn mainly from locally available signals, and no clear biological equivalent of backpropagation is known.

:::

::: fragment

The clearest example is the [transformer architecture]{.underline}: achieved a big boost in performance obtained _precisely by abandoning some of the most brain-inspired ideas_ (recurrence and convolution).

:::



## Transformer architecture

:::::: columns

:::: {.column width="50%"}

::: {style="font-size: 80%;"}

From a neuroscience perspective, several aspects are strikingly un-brain-like, and hard to justify except by their empirical success:

- [Parallel processing of all [_tokens_](https://platform.openai.com/tokenizer)]{.underline} ($\approx$ words) in a single feedforward pass, rather than sequentially
- [Global pairwise token interactions]{.underline}: every token can interact directly with every other token
- [Multi-head attention]{.underline}: multiple parallel attention mechanisms operate at once
- [Positional encoding]{.underline}: sequence order must be added explicitly because the architecture has no built-in notion of order

:::

::::

:::: {.column width="50%"}

![](img/ModalNet-21.png){fig-align="center" width="80%"}

::::

::::::


## 

####  Positional encoding in transformers

:::::: columns

:::: {.column width="50%"}

::: {style="font-size: 70%;"}

:::: nonincremental

- Tokens are first represented using a _one-hot_ code. 

- They are then mapped onto lower-dimensional dense embeddings, vectors[^vector] of numbers that capture aspects of word meaning, so that related words lie close together in the embedding space.

::::

- Because transformers process all tokens simultaneously, they have no built-in sense of sequence. To encode word order, a position vector is added to each embedding.

- [This is a strikingly un-brain-like solution: rather than sequence emerging from dynamics or recurrence, order is imposed by adding an explicit numerical code for position.]{style="color:#198754"}

[^vector]: A vector is simply an ordered list of numbers. An $n$-dimensional vector defines a position in an $n$-dimensional space.

:::

:::::: fragment



::: {.cell layout-align="center"}
::: {.cell-output-display}
![Positional encoding matrix for a text of 50 words, and an internal embedding dimensions of 100.](AI_slides_files/figure-revealjs/post-1.png){fig-align='center' width=288}
:::
:::



::::::

:::: 
 
:::: {.column width="50%"}

![](img/ModalNet-21.png){fig-align="center" width="80%"}

::::

::::::

## How do transformers learn?

Like earlier neural networks, transformers learn by adjusting millions or billions of parameters through **backpropagation**.

- During training, the model is shown huge amounts of text and asked to predict the next token
- When its prediction is wrong, the error is computed and propagated backward through the network
- This gradually adjusts the weights so that future predictions improve
- Although first developed for sequence-to-sequence tasks such as translation, this training objective turned out to support many other abilities too


## After 2017

- **Transformers rapidly scaled up**: more data, more compute, and many more parameters
- **Multimodal models** extended the same general approach beyond text, combining language with images, audio, and other inputs
- **ChatGPT** made this technology widely visible when it was released to the public in November 2022





## Practical session: calling language models from R

Today we will use the [**ellmer**](https://ellmer.tidyverse.org/) package in R to call large language models from code.

- `ellmer` provides a simple interface to a wide range of model providers
- It lets us send prompts, receive model outputs and process them directly in R


