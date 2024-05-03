import os
import pickle
import re
from multiprocessing import cpu_count
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from huggingface_hub import login
from peft import PeftConfig, PeftModel, LoraConfig
from trl import DPOTrainer
from transformers import BitsAndBytesConfig

def main():
    # Read dataset from pickle file
    with open("data/custom_pref_data_70b.pickle", "rb") as f:
        raw_datasets = pickle.load(f)

    # Load Tokenizer
    model_id = "alignment-handbook/zephyr-7b-sft-lora"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    # Apply Chat-Template
    def apply_chat_template(example, tokenizer, assistant_prefix="\n"):
        def _strip_prefix(s, pattern):
            return re.sub(f"^{re.escape(pattern)}", "", s)

        if all(k in example.keys() for k in ("chosen", "rejected")):
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
        return example

    column_names = list(raw_datasets["train"].features)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=cpu_count(),
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Load SFT Model
    peft_config = PeftConfig.from_pretrained(model_id)

    # Define quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    # Load the base model
    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=False,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)

    # Load base model + SFT adapter weights
    model = PeftModel.from_pretrained(base_model, model_id)

    # Define DPO Trainer
    output_dir = 'data/dpo_model-70b'
    training_args = TrainingArguments(
        bf16=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        hub_model_id="zephyr-70b-dpo-qlora",
        learning_rate=5.0e-6,
        log_level="info",
        logging_steps=10,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        optim="paged_adamw_32bit",
        output_dir=output_dir, 
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=8,  
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        seed=42,
        warmup_ratio=0.1,
    )

    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",  "up_proj",  "down_proj"],
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,
        model_init_kwargs=None,
        ref_model_init_kwargs=None,
        args=training_args,
        beta=0.01,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
        peft_config=peft_config,
        loss_type='sigmoid',
    )

    # Training
    torch.cuda.empty_cache()
    print('Everything loaded. Starting training!')
    train_result = trainer.train()

    # Save Training
    metrics = train_result.metrics
    max_train_samples = len(raw_datasets["train"])
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print('All done :D')

if __name__ == "__main__":
    main()
