from transformers import AutoTokenizer
from scripts.data_prep import load_text_data, tokenize_data
from scripts.model_setup import load_model_and_tokenizer
from scripts.training import train_model
from scripts.evaluation import generate_text

# Step 1: Prepare data
print("——————————")
print("Step 1: Preparing data...")
data_folder = "data/original_text"
dataset = load_text_data(data_folder)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenized_data = tokenize_data(dataset, tokenizer)
tokenized_data.save_to_disk("data/tokenized_data")
print("Data preparation complete.")

# Step 2: Load model and tokenizer
print("——————————")
print("Step 2: Loading model and tokenizer...")
model, tokenizer = load_model_and_tokenizer()
print("Model and tokenizer loaded.")

# Step 3: Finetune model
print("——————————")
print("Step 3: Finetuning model...")
train_model(model, tokenizer, tokenized_data_path="data/tokenized_data")
print("Model finetuning complete.")

# Step 4: Evaluate model
print("——————————")
print("Step 4: Evaluating model...")
# prompt = "Write me an essay about the possibility of AGI ruin and how we can live in the face of mortality"
prompt = "In another universe, sunlight drips through an open window. I am lying on the floor in a small apartment. I am five years old. Nothing bad has happened to me. In this version of the story, my life is still a blank slate, and I can control what happens like one of those Choose-Your-Own-Adventure video games. The critical events that will go on to change my life—the move to America, the class switch in high school, the fateful LinkedIn DM—have yet to take place. Nothing bad has happened, nothing good has happened, and I was still naïve enough to believe I could shape the future to my will."
generated_text = generate_text(model, tokenizer, prompt)
print("Generated text:")
print(generated_text)

print("——————————")