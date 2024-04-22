import os
import pickle
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoTokenizer
import re
from multiprocessing import cpu_count
from peft import PeftConfig
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from huggingface_hub import login

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] ="expandable_segments:True"

    hugging_face_access_token = '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'
    # login(token="your_access_token")

    # Read dataset from pickle file
    with open("data/custom_pref_data_7b.pickle", "rb") as f:
        raw_datasets = pickle.load(f)

    # indices = range(0,100)

    # dataset_dict = {"train": raw_datasets["train"].select(indices),
    #             "test": raw_datasets["test"].select(indices)}

    # raw_datasets = DatasetDict(dataset_dict)

    # Load Tokenizer
    model_id = "alignment-handbook/zephyr-7b-sft-lora"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hugging_face_access_token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Truncate from left to ensure we don't lose labels in final turn
    tokenizer.truncation_side = "left"

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


    # Apply Chat-Template
    def apply_chat_template(example, tokenizer, assistant_prefix="<|assistant|>\n"):
        def _strip_prefix(s, pattern):
            # Use re.escape to escape any special characters in the pattern
            return re.sub(f"^{re.escape(pattern)}", "", s)

        if all(k in example.keys() for k in ("chosen", "rejected")):
                # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
                prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
                # Insert system message
                if example["chosen"][0]["role"] != "system":
                    prompt_messages.insert(0, {"role": "system", "content": ""})
                else:
                    prompt_messages.insert(0, example["chosen"][0])
                # TODO: handle case where chosen/rejected also have system messages
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

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )


    # Load SFT Model
    peft_config = PeftConfig.from_pretrained(model_id, token = hugging_face_access_token)

    # specify how to quantize the model
    quantization_config = BitsAndBytesConfig(
                load_in_8bit=False,
                load_in_4bit=True,
                # bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_type="fp4",
                # bnb_4bit_compute_dtype=torch.float16, # ggf. zu float16 wechseln
                bnb_4bit_compute_dtype=torch.bfloat16, # ggf. zu float16 wechseln
    )
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

    # Step 1: load the base model (Mistral-7B in our case) in 4-bit
    model_kwargs = dict(
        # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype="auto",
        use_cache=False,  # set to False as we're going to use gradient checkpointing
        device_map=device_map,
        quantization_config=quantization_config,
    )
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs, token = hugging_face_access_token)

    # Step 2: load base model + SFT adapter weights
    # notice that only the adapter weights are trainable!
    model = PeftModel.from_pretrained(base_model, model_id, token = hugging_face_access_token)



    # Define DPO Trainer
    from trl import DPOTrainer
    from peft import LoraConfig
    from transformers import TrainingArguments

    # path where the Trainer will save its checkpoints and logs
    # output_dir = 'data/zephyr-7b-dpo-lora'
    output_dir = 'data/dpo_model-7b'

    # based on config
    training_args = TrainingArguments(
        bf16=True,
        # fp16=True,
        # beta=0.01,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        hub_model_id="zephyr-7b-dpo-qlora",
        learning_rate=5.0e-6,
        log_level="info",
        logging_steps=10,
        lr_scheduler_type="cosine",
        # max_length=1024,
        # max_prompt_length=512,
        num_train_epochs=1,
        optim="paged_adamw_32bit",
        output_dir=output_dir,  # It is handy to append `hub_model_revision` to keep track of your local experiments
        per_device_train_batch_size=4,  # original: 4
        per_device_eval_batch_size=8,   # original: 8
        # push_to_hub=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        seed=42,
        warmup_ratio=0.1,
    )

    # based on the recipe: https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/dpo/config_qlora.yaml
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
            # beta=training_args.beta,
            beta=0.01,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            tokenizer=tokenizer,
            # max_length=training_args.max_length,
            max_length=1024,
            # max_prompt_length=training_args.max_prompt_length,
            max_prompt_length=512,
            peft_config=peft_config,
            # loss_type=training_args.loss_type,
            loss_type='sigmoid',
        )


    # Training
    torch.cuda.empty_cache()
    print('Everything loaded. Starting training!')
    train_result = trainer.train()

    # Save Training
    metrics = train_result.metrics
    # max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(raw_datasets["train"])
    max_train_samples = len(raw_datasets["train"])
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print('All done :D')



if __name__ == "__main__":
    main()