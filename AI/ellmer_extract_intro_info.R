library(ellmer)
library(readr)
library(dplyr)
# library(purrr)
library(jsonlite)

# Load the dataset
intro_msgs <- read_csv("ellmer_forum_intro_dataset.csv", show_col_types = FALSE)

# Create a chat object (adjust the model/provider to match your setup)
chat <- chat_openai(
  model = "gpt-4.1-mini",
  system_prompt = paste(
    "You extract structured information from short self-introduction messages.",
    "Return ONLY valid JSON with exactly these keys:",
    '{"name":"...","age":0}',
    "If you are uncertain, make the best single guess from the message."
  )
)

results <- vector("list", nrow(intro_msgs))

for (i in seq_len(nrow(intro_msgs))) {

  prompt <- paste(
    "Extract the person's name and age from this forum introduction.",
    "Return JSON only.",
    "",
    intro_msgs$message[i]
  )

  raw <- chat$chat(prompt)
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
