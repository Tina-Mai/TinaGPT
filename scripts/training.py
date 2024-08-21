from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datasets import load_from_disk

def train_model(model, tokenizer, tokenized_data_path, output_dir="./models/llama-finetuned"):
    tokenized_data = load_from_disk(tokenized_data_path)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=False,
        save_total_limit=3,
        logging_steps=1,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)