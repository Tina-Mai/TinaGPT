# handles evaluation of the model
# TODO: doesn't work if ran on its own yet (need to fix load_finetuned_model)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_finetuned_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_finetuned_model("./models/llama-finetuned")
    prompt = "In another universe, sunlight drips through an open window. I am lying on the floor in a small apartment. I am five years old. Nothing bad has happened to me. In this version of the story, my life is still a blank slate, and I can control what happens like one of those Choose-Your-Own-Adventure video games. The critical events that will go on to change my life—the move to America, the class switch in high school, the fateful LinkedIn DM—have yet to take place. Nothing bad has happened, nothing good has happened, and I was still naïve enough to believe I could shape the future to my will."
    generated_text = generate_text(model, tokenizer, prompt)
    print(generated_text)