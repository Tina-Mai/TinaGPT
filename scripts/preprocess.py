from datasets import Dataset

# example: loading data from a text file
data = Dataset.from_text('data/original_text/essay_3_4.txt')

tokenizer = LLaMATokenizer.from_pretrained('meta-llama/LLaMA-3B')

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_data = data.map(tokenize_function, batched=True, remove_columns=["text"])