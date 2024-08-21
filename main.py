from transformers import AutoTokenizer
from scripts.data_prep import load_text_data, tokenize_data
from scripts.model_setup import load_model_and_tokenizer
from scripts.training import train_model
from scripts.evaluation import generate_text

# step 1: prepare data
print("Step 1: Preparing data...")
data_folder = "data/original_text"
dataset = load_text_data(data_folder)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenized_data = tokenize_data(dataset, tokenizer)
tokenized_data.save_to_disk("data/tokenized_data")
print("Data preparation complete.")

# step 2: load model and tokenizer
print("Step 2: Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer()
print("Model and tokenizer loaded.")

# step 3: finetune model
print("Step 3: Finetuning model...")
train_model(model, tokenizer, tokenized_data_path="data/tokenized_data")
print("Model finetuning complete.")

# step 4: evaluate model
print("Step 4: Evaluating model...")
prompt = "Write me an essay about the possibility of AGI ruin and how we can live in the face of mortality"
generated_text = generate_text(model, tokenizer, prompt)
print("Generated text:")
print(generated_text)
