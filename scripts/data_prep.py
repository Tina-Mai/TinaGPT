# handles loading and tokenizing the data

import os
from datasets import Dataset
from transformers import AutoTokenizer


def load_text_data(data_folder):
    texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(data_folder, filename), 'r') as file:
                texts.append(file.read())
    return Dataset.from_dict({"text": texts})

def tokenize_data(dataset, tokenizer, max_length=256):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])


if __name__ == "__main__":
    data_folder = "data/original_text"
    dataset = load_text_data(data_folder)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

    # set the padding token to the eos_token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_data = tokenize_data(dataset, tokenizer)
    tokenized_data.save_to_disk("data/tokenized_data")
