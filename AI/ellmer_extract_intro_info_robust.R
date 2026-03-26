library(ellmer)
library(readr)
library(dplyr)
library(purrr)
library(jsonlite)

intro_msgs <- read_csv("ellmer_forum_intro_dataset.csv", show_col_types = FALSE)

chat <- chat_openai(
  model = "gpt-4.1-mini",
  system_prompt = paste(
    "You extract structured data from casual online introductions.",
    "Return ONLY compact JSON with keys name and age.",
    'Example: {"name":"Alex","age":42}'
  )
)

safe_extract <- function(text) {
  raw <- chat$chat(
    paste(
      "Extract the name and age from this message.",
      "Return JSON only with keys name and age.",
      "",
      text
    )
  )

  out <- tryCatch(
    fromJSON(raw),
    error = function(e) list(name = NA_character_, age = NA_integer_)
  )

  tibble(
    name = as.character(out$name),
    age = as.integer(out$age)
  )
}

extracted <- map_dfr(intro_msgs$message, safe_extract)
final <- bind_cols(intro_msgs, extracted)

print(final)
