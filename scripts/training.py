# handles training the model

from transformers import Trainer, TrainingArguments
from scripts.model_setup import load_model_and_tokenizer
from datasets import load_from_disk

def train_model(model, tokenizer, tokenized_data_path, output_dir="./models/llama-finetuned"):
    tokenized_data = load_from_disk(tokenized_data_path)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    train_model(model, tokenizer, tokenized_data_path="data/tokenized_data")