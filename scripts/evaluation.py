# handles evaluation of the model

from scripts.model_setup import load_model_and_tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_text(model, tokenizer, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(model_name="./llama-finetuned")
    prompt = "Write me an essay about the possibility of AGI ruin and how we can live in the face of mortality"
    generated_text = generate_text(model, tokenizer, prompt)
    print(generated_text)
