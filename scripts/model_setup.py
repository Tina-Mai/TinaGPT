# handles loading and saving the model and tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name='meta-llama/Meta-Llama-3-8B'):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model.save_pretrained("models/llama_base")
    tokenizer.save_pretrained("models/llama_base")