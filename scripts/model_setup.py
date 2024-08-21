# handles loading and saving the model and tokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel, PeftConfig

model_id = "meta-llama/Meta-Llama-3-8B"
peft_model_id = "ybelkada/opt-350m-lora"

def load_model_and_tokenizer(model_name=model_id, peft_model_name=peft_model_id):
    # load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )
    
    # prepare the model for training
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # load the PEFT adapter
    model = PeftModel.from_pretrained(model, peft_model_name)
    
    # ensure all adapter layers are trainable
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model.save_pretrained("models/llama_base")
    tokenizer.save_pretrained("models/llama_base")
