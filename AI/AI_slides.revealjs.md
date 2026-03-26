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

:::::::: columns
::::: {.column width="60%"}
   

::: {style="font-size: 70%;"}
-   **Donald Hebb (1949)** *"Neurons that fire together, wire together"*
:::

<!-- Learning as strengthening of synaptic connections between co-active neurons -->

 

::: {style="font-size: 70%;"}
-   **Frank Rosenblatt (1962)** build the **Perceptron**: a single layer network that can learn weights from data
:::
:::::

:::: {.column width="40%"}
::: fragment
![](img/retina-perceptron.png){fig-align="center" width="100%"}

![](img/rosenblatt.jpg){fig-align="center" width="50%"}
:::
::::
::::::::

## Limitations

<!-- {.smaller} -->

<!-- #### The XOR problem & Minsky–Papert (1969) -->

 

:::::::::::: columns
::::::: {.column width="40%"}
:::::: {style="font-size: 70%;"}
::: {.fragment fragment-index="1"}
*Perceptron Convergence Theorem* — if a *linear* solution exists, the perceptron is guaranteed to find it.
:::

 

::: {.fragment fragment-index="2"}
The bad news: many real problems are **not linearly separable** — the classic case is XOR:

| x₁  | x₂  | XOR |
|-----|-----|-----|
| 0   | 0   | 0   |
| 0   | 1   | 1   |
| 1   | 0   | 1   |
| 1   | 1   | 0   |

No single straight line can separate the 1s from the 0s.
:::

 

::: {.fragment fragment-index="3"}
[Minsky & Papert (1969) formalised these limitations — setting off the first **AI Winter**.]{style="color:#198754"}
:::
::::::
:::::::

:::: {.column width="30%"}
::: {.fragment fragment-index="2"}


::: {.cell}
::: {.cell-output-display}
![](AI_slides_files/figure-revealjs/xor-1.png){width=336}
:::
:::


:::
::::

:::: {.column width="30%"}
::: {.fragment fragment-index="3"}
![](img/perceptron_book_cover.jpg){fig-align="center" width="100%"}
:::
::::
::::::::::::

## Single and multi-layer perceptrons

:::::: columns
::: {.column width="50%"}
![](img/single.png){fig-align="center" width="80%"}
:::

:::: {.column width="50%"}
::: fragment
![](img/multi.png){fig-align="center" width="80%"}
:::
::::
::::::

## The first "AI winter" (1970-1980)

:::::: nonincremental
::: {.fragment fragment-index="1"}
-   Period of severely reduced funding and interest in neural network research, due to unmet expectations.
:::

::: {.fragment fragment-index="2"}
-   Funding and energy moved toward **symbolic AI**, focused on using high-level representations and logic to model human intelligence.
:::

::: {.fragment fragment-index="3"}
-   Minsky and Papert, among others, were proponents of symbolic AI approaches and argued that neural networks would fail to scale up beyond toy problems.
:::
::::::

 

::: {.fragment fragment-index="4"}
A second AI winter in the late 80s, when it became clear that many buyers and investors expected far more adaptability and robustness than what symbolic AI systems could provide.
:::

 

::: {.fragment fragment-index="5"}
[***Typical*** **AI winter pattern: not "no progress", but expectations rising much faster than capabilities.**]{style="color:#198754"}
:::

## Key milestones in neural network AI {.smaller .scrollable background-color="#202A30"}



::: {.cell layout-align="center"}
::: {.cell-output-display}
![](AI_slides_files/figure-revealjs/timeline-1.png){fig-align='center' width=1344}
:::
:::



-   1982: **Backpropagation algorithm**: scalable training of multilayer perceptrons revives interest in neural networks

-   1989 **Convolutional neural network (CNN)**: spatial convolutions inspired by the hierarchical organisation of the visual system

-   1997 **Long short-term memory (LSTM)**: recurrent network that can learn long-range dependencies in sequences, inspired by recurrence in brain circuits

-   2012 **Deep learning**: AlexNet wins the ImageNet competition by a large margin, sparking major industry interest

-   2013 **word2vec** dense distributed word embeddings learned from huge text corpora, allowing to represent semantics with list of numbers (not unlike the brain)

-   2017 *"Attention is all you need"* paper introduces the **transformer** architecture, which underlies modern large language models (GPT, Claude, Gemini, etc.)

## AlexNet & the ImageNet competition

 

::::: columns
::: {.column width="70%"}
![Krizhevsky, Sutskever, & Hinton (2012)](img/Krizhevsky2012_architecture.png){fig-align="center" width="100%"}
:::

::: {.column width="30%"}
![ImageNet contains over 14 million images categorized into 1000 classes.](img/imagenet.webp){fig-align="center" width="100%"}
:::
:::::

 

::: fragment
The success of AlexNet was due also to recent developments in availability of large scale labelled datasets and general purpose GPU computing.
:::

## How brain-like are neural networks? {.scrollable}

 

Many influential ideas in neural networks were borrowed from biology:

::: {style="font-size: 70%;"}
-   The artificial neuron as a basic computational unit in a network

-   Early learning algorithms (e.g. the perceptron) were inspired by the strengthening of connections between co-active neurons (Hebb rule)

-   Information is stored in the strength of synaptic connections, a central idea in neuroscience

-   Convolutional networks were heavily inspired by biological vision (local receptive fields, hierarchical feature detectors)

-   Early approaches to language and sequence learning relied on *recurrence*, echoing recurrent connectivity in the brain

-   Semantic concepts can be represented in a distributed way: both artificial and biological networks encode information as patterns of activity across many units
:::

   

::: fragment
[Over time, neural networks increasingly diverged from biology, replacing biological realism with engineering abstractions.]{style="color:#198754"}
:::

   

::: fragment
A first example is [backpropagation]{.underline}: it uses the chain rule to propagate errors backward and update connection weights during learning. By contrast, biological neurons are thought to learn mainly from locally available signals, and no clear biological equivalent of backpropagation is known.
:::

   

::: fragment
The clearest example is the [transformer architecture]{.underline}: achieved a big boost in performance obtained *precisely by abandoning some of the most brain-inspired ideas* (recurrence and convolution).
:::

## Transformer architecture

:::::: columns
:::: {.column width="50%"}
::: {style="font-size: 80%;"}
From a neuroscience perspective, several aspects are strikingly un-brain-like, and hard to justify except by their empirical success:

-   [Parallel processing of all [*tokens*](https://platform.openai.com/tokenizer)]{.underline} ($\approx$ words) in a single feedforward pass, rather than sequentially
-   [Global pairwise token interactions]{.underline}: every token can interact directly with every other token
-   [Multi-head attention]{.underline}: multiple parallel attention mechanisms operate at once
-   [Positional encoding]{.underline}: sequence order must be added explicitly because the architecture has no built-in notion of order
:::
::::

::: {.column width="50%"}
![](img/ModalNet-21.png){fig-align="center" width="80%"}
:::
::::::

## 

#### Positional encoding in transformers

:::::::: columns
:::::: {.column width="50%"}
:::: {style="font-size: 70%;"}
::: nonincremental
-   Tokens are first represented using a *one-hot* code.

-   They are then mapped onto lower-dimensional dense embeddings, vectors[^1] of numbers that capture aspects of word meaning, so that related words lie close together in the embedding space.
:::

-   Because transformers process all tokens simultaneously, they have no built-in sense of sequence. To encode word order, a position vector is added to each embedding.

-   [This is a strikingly un-brain-like solution: rather than sequence emerging from dynamics or recurrence, order is imposed by adding an explicit numerical code for position.]{style="color:#198754"}
::::

::: fragment


::: {.cell layout-align="center"}
::: {.cell-output-display}
![Positional encoding matrix for a text of 50 words, and an internal embedding dimensions of 100.](AI_slides_files/figure-revealjs/post-1.png){fig-align='center' width=288}
:::
:::


:::
::::::

::: {.column width="50%"}
![](img/ModalNet-21.png){fig-align="center" width="80%"}
:::
::::::::

[^1]: A vector is simply an ordered list of numbers. An $n$-dimensional vector defines a position in an $n$-dimensional space.

## 

Visual explainer: <https://poloclub.github.io/transformer-explainer/>




```{=html}
<iframe width="1050" height="600" src="https://poloclub.github.io/transformer-explainer/"></iframe>
```




## How do transformers learn?

Like earlier neural networks, transformers learn by adjusting millions or billions of parameters through **backpropagation**.

-   During training, the model is shown huge amounts of text and asked to predict the next token
-   When its prediction is wrong, the error is computed and propagated backward through the network
-   This gradually adjusts the weights so that future predictions improve

 

::: fragment
Although first developed for sequence-to-sequence tasks such as translation, this training objective turned out to support other abilities too.
:::

## After 2017

-   **Transformers rapidly scaled up**: more data, more compute, and many more parameters
-   **Multimodal models** extended the same general approach beyond text, combining language with images, audio, and other inputs
-   **ChatGPT** made this technology widely visible when it was released to the public in November 2022

<!-- ## Ethics of modern AI -->

<!-- - My view: nothing inherently wrong with the technology, but the investment bubble likely to have negative effects for environment, wealth inequality, etc. -->

<!-- - AI misalignment: models have 'no wants', but if not regulated can definitely have unintended, negative consequences -->

<!-- - Even worse when given agentic capabilities (e.g. OpenClaw) -->

## Limits of large language models

 

::: {layout="[[1,1]]"}

![[link](https://www.pnas.org/doi/10.1073/pnas.2322420121)](img/embers.png){fig-align="center" width="70%"}

![](img/pnas.2322420121fig05.jpg)


:::

<!-- _"Stochastic parrots" shaped by the problem they are trained to solve_ -->

## Ethics of AI

::: {layout="[[1,1, 1]]"}
![](img/Empire_of_AI_book_cover.jpg)

![[FT article link](https://www.ft.com/content/60e2a900-8999-46cc-8107-4f468f442aae)](img/FTwomen.png)


![[link](https://theshamblog.com/an-ai-agent-published-a-hit-piece-on-me/)](img/shamblog_AImisalign.png)


:::

:::: {style="font-size: 70%;"}

Critical AI literacy resources by Olivia Guest: [https://olivia.science/ai/](https://olivia.science/ai/)

::::




# Practical session: calling language models from R

## 

Today we will use the [**ellmer**](https://ellmer.tidyverse.org/) package in R to call large language models from code.

-   `ellmer` provides a simple interface to a wide range of model providers
-   It lets us send prompts, receive model outputs and process them directly in R

##  {.scrollable background-color="#202A30"}

#### Steps to create a Gemini API key and adding it to R environment

1.  Go to: <https://ai.google.dev/aistudio>

2.  Click **GET API KEY** in the top bar.

3.  Sign in with a Google account, or create one if needed.

4.  After logging in, you should be taken to: <https://aistudio.google.com/api-keys>

5.  Click **Create API key**. \newline After entering a name, a new API key will appear in the list on that page. Click the 'copy' key icon, then return to RStudio.

6.  In RStudio, run the following command in the console: `usethis::edit_r_environ()` This will open your .Renviron file in the editor. Add a line that starts like this: `GEMINI_API_KEY=<paste API key here>` making sure to paste the API key you created above after the equal sign. Then save and close the file.

7.  Restart R (you can do this by pressing Ctrl + Shift + F10, or by clicking Session \> Restart R).

8.  Testing that it works After completing these steps, you can test whether everything is working by running:

::: fragment


::: {.cell}

```{.r .cell-code}
library(ellmer)

# start a chat
chat <- chat_google_gemini()

# test a prompt
chat$chat("Explain why overfitting in machine learning is a problem.")
```
:::


:::

##  {.scrollable}



::: {.cell}

```{.r .cell-code}
library(ellmer)

# chat <- chat_google_gemini("You are a terse assistant.")
chat <- chat_openai("You are a terse assistant.") 

chat$chat("What is the capital of Italy?")
```

::: {.cell-output .cell-output-stdout}

```
Rome is the capital of Italy.
```


:::

```{.r .cell-code}
# the client is stateful, so this continues the conversation
chat$chat("What is its most famous landmark?")
```

::: {.cell-output .cell-output-stdout}

```
The Colosseum is Rome's most famous landmark.
```


:::
:::



   

::: fragment


::: {.cell}

```{.r .cell-code}
chat
```

::: {.cell-output .cell-output-stdout}

```
<Chat OpenAI/gpt-4.1 turns=5 tokens=70/18 $0.00>
── system [0] ──────────────────────────────────────────────────────────────────
You are a terse assistant.
── user [24] ───────────────────────────────────────────────────────────────────
What is the capital of Italy?
── assistant [7] ───────────────────────────────────────────────────────────────
Rome is the capital of Italy.
── user [15] ───────────────────────────────────────────────────────────────────
What is its most famous landmark?
── assistant [11] ──────────────────────────────────────────────────────────────
The Colosseum is Rome's most famous landmark.
```


:::
:::


:::

## Anatomy of a conversation

-   Each interaction is a pair of user and assistant turns, corresponding to a HTTP request and response.

-   Messages are in JSON (JavaScript Object Notation) format

-   The API server is *stateless* (does not store anything about the conversation), even though conversation are *statefull*.

<!-- ```{r} -->

<!-- httr2::with_verbosity(chat$chat("What is its most famous landmark?"), verbosity=2) -->

<!-- httr2::local_verbosity(2, env = caller_env()) -->

<!-- ``` -->

##  {.scrollable}

We can see what is going on under the hood using `options(httr2_verbosity = 2)`

   

User request

```         
 {
   "contents": [
     {
       "role": "user",
       "parts": [
         {
           "text": "What is the capital of Italy?"
         }
       ]
     }
   ],
   "systemInstruction": {
     "parts": {
       "text": "You are a terse assistant."
     }
   }
}
```

##  {.scrollable}

API server response

```         
type: message
data: {
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "Rome"
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 14,
    "candidatesTokenCount": 1,
    "totalTokenCount": 35,
    "promptTokensDetails": [
      {
        "modality": "TEXT",
        "tokenCount": 14
      }
    ],
    "thoughtsTokenCount": 20
  },
  "modelVersion": "gemini-2.5-flash",
  "responseId": "X0rEaeDNCpCuxN8P0YmZiAE"
}
```

## Working with 'unstructured' data {.scrollable}




::: {.cell}

```{.r .cell-code}
recipe <- "
  In a large bowl, cream together 1 cup of softened unsalted butter and ½ cup
  of white sugar until smooth. Beat in 1 egg and 1 teaspoon of vanilla extract.
  Gradually stir in 2 cups of all-purpose flour until the dough forms. Finally,
  fold in 1 cup of semisweet chocolate chips. Drop spoonfuls of dough onto an
  ungreased baking sheet and bake at 350°F (175°C) for 10-12 minutes, or until
  the edges are lightly browned. Let the cookies cool on the baking sheet for
  a few minutes before transferring to a wire rack to cool completely. Enjoy!
"

chat <- chat_openai("
  The user input contains a recipe. Extract a list of ingredients and
  return it in table format."
)
chat$chat(recipe)
```

::: {.cell-output .cell-output-stdout}

```
| Ingredient                      | Amount         |
|----------------------------------|---------------|
| Unsalted butter (softened)       | 1 cup         |
| White sugar                      | ½ cup         |
| Egg                              | 1             |
| Vanilla extract                  | 1 teaspoon    |
| All-purpose flour                | 2 cups        |
| Semisweet chocolate chips        | 1 cup         |
```


:::
:::




## 

Another example of unstructured data



::: {.cell}
::: {.cell-output .cell-output-stdout}

```
# A tibble: 40 × 2
   message_id message                                                           
   <chr>      <chr>                                                             
 1 msg_001    Hey everyone — I'm Nina, and I turned 24 this spring.             
 2 msg_002    Most folks online know me as Mateo. I'm 31 years old.             
 3 msg_003    Hi all! Priya here, 22 and excited to join this server.           
 4 msg_004    The username's Omar, but you can just call me Omar — I'm 46.      
 5 msg_005    Hello from Sophie. I celebrated number 29 a couple months ago.    
 6 msg_006    Name's Diego. Been around for 38 years now.                       
 7 msg_007    What's up, I'm Aisha, age 26, joining from Nairobi.               
 8 msg_008    You can call me Benji. Just hit 33 last month.                    
 9 msg_009    Hey group, this is Elena checking in — 41 years old today, actual…
10 msg_010    Jordan here. Nineteen and mostly here to learn.                   
# ℹ 30 more rows
```


:::
:::



_Can you write code to extract name and age information from the introduction messages?_


## {.scrollable}



::: {.cell}

```{.r .cell-code}
library(jsonlite)

chat <- chat_openai(
  system_prompt = paste(
    "You extract structured information from short self-introduction messages.",
    "Return ONLY valid JSON with exactly these keys:",
    '{"name":"...","age":0}'
  )
)

results <- vector("list", nrow(intro_msgs))

for (i in 1:nrow(intro_msgs)) {

  prompt <- paste(
    "Extract the person's name and age from this forum introduction.",
    "Return JSON only.",
    "",
    intro_msgs$message[i]
  )

  raw <- chat$chat(prompt,echo = "none")
  parsed <- fromJSON(raw)

  results[[i]] <- tibble(
    message_id = intro_msgs$message_id[i],
    message = intro_msgs$message[i],
    name = parsed$name,
    age = parsed$age
  )
}

extracted <- bind_rows(results)

print(extracted)
```

::: {.cell-output .cell-output-stdout}

```
# A tibble: 40 × 4
   message_id message                                                name    age
   <chr>      <chr>                                                  <chr> <int>
 1 msg_001    Hey everyone — I'm Nina, and I turned 24 this spring.  Nina     24
 2 msg_002    Most folks online know me as Mateo. I'm 31 years old.  Mateo    31
 3 msg_003    Hi all! Priya here, 22 and excited to join this serve… Priya    22
 4 msg_004    The username's Omar, but you can just call me Omar — … Omar     46
 5 msg_005    Hello from Sophie. I celebrated number 29 a couple mo… Soph…    29
 6 msg_006    Name's Diego. Been around for 38 years now.            Diego    38
 7 msg_007    What's up, I'm Aisha, age 26, joining from Nairobi.    Aisha    26
 8 msg_008    You can call me Benji. Just hit 33 last month.         Benji    33
 9 msg_009    Hey group, this is Elena checking in — 41 years old t… Elena    41
10 msg_010    Jordan here. Nineteen and mostly here to learn.        Jord…    19
# ℹ 30 more rows
```


:::
:::



# Where to go from here

## What have you learned?

-   **R Programming Fundamentals**\
    Data types, data structures, control flow, and custom functions\
-   **Data Importing & Cleaning**\
    Using `tidyverse` packages (`readr`, `dplyr`, `tidyr`)\
-   **Data Visualization**\
    Base R plotting and ggplot2 basics\
-   **Machine Learning Foundations**\
    Supervised (regression, classification) and unsupervised (clustering, GMMs)\
-   **Reproducible Reporting**\
    R Markdown for transparent, documented analysis

------------------------------------------------------------------------

## Where to go next?

## Where to go next?

:::: {.fragment fragment-index="1"}
**Further study:**

::: nonincremental
-   [***R for Data Science***](https://r4ds.had.co.nz/) by Hadley Wickham (practical, applied DS)\
-   [***Statistical Rethinking***](https://xcelab.net/rm/) by Richard McElreath (Bayesian/statistical modeling, great for academic paths)
:::
::::

:::: {.fragment fragment-index="2"}
::: nonincremental
-   [***Statistics for Psychology using R: a linear models perspective***](https://www.mheducation.co.uk/statistics-for-psychology-using-r-a-linear-models-perspective-9780335252626-emea-group)\
    by Alasdair Clarke & myself.
:::
::::

:::::::::: columns
::::::: {.column width="70%"}
:::: {.fragment fragment-index="3"}
::: nonincremental
-   **Practice, Practice, Practice**
    -   Explore Kaggle datasets or other open data portals\
    -   Join hackathons or coding challenges\
:::
::::

:::: {.fragment fragment-index="4"}
::: nonincremental
-   **Self-directed Projects**
    -   Pick a dataset in your area of interest (e.g. OSF)\
    -   Try cleaning, analyzing, and visualizing it from scratch\
:::
::::
:::::::

:::: {.column width="30%"}
::: {.fragment fragment-index="2"}
![](img/book_cover.png){fig-align="center" width="75%"}
:::
::::
::::::::::

------------------------------------------------------------------------

## Get out there & connect!

-   **Data Volunteer Opportunities**
    -   [DataKind UK](https://www.datakind.org.uk/resources/data-volunteering-development)
        -   Projects supporting nonprofits, social causes\
        -   Great place to learn while contributing\
-   **Networking & Communities**
    -   [R-Ladies](https://www.meetup.com/pro/rladies/): Inclusive community for R users\
    -   [Black in Data](https://www.blackindata.co.uk/): UK-based collective for representation in data fields\
    -   [Women in Data](https://womenindata.co.uk/): Community events, mentorship, and resources




