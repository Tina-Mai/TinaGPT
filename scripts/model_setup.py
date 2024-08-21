# handles loading and saving the model and tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model_and_tokenizer(model_name='meta-llama/Meta-Llama-3-8B'):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # load in 8-bit quantization
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model.save_pretrained("models/llama_base")
    tokenizer.save_pretrained("models/llama_base")
