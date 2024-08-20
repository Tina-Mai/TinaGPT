from transformers import AutoTokenizer
from scripts.data_prep import load_text_data, tokenize_data
from scripts.model_setup import load_model_and_tokenizer
from scripts.training import train_model
from scripts.evaluation import generate_text

# step 1: prepare data
data_folder = "data/original_text"
dataset = load_text_data(data_folder)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
tokenized_data = tokenize_data(dataset, tokenizer)
tokenized_data.save_to_disk("data/tokenized_data")

# step 2: load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# step 3: finetune model
train_model(model, tokenizer, tokenized_data_path="data/tokenized_data")

# step 4: evaluate model
prompt = "Your test prompt here"
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)