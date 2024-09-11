import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login
import wandb
import pandas as pd

# Output folder
output_dir = "./results"

# No of epochs
num_train_epochs =1

# No change params
use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant = True, "float16", "nf4", False # To quantization
lora_r, lora_alpha, lora_dropout = 64, 16, 0.1
fp16, bf16 =  False, False
per_device_train_batch_size, per_device_eval_batch_size = 4, 4
gradient_accumulation_steps, gradient_checkpointing, max_grad_norm = 1, True, 0.3
learning_rate, weight_decay, optim = 2e-4, 0.001, "paged_adamw_32bit"
lr_scheduler_type, max_steps, warmup_ratio = "constant", -1, 0.03
group_by_length, save_steps, logging_steps = True, 0, 25
max_seq_length, packing, device_map = None, False, {"": 0}

# Model from Hugging Face hub
base_model = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
# Fine-tuned model
new_model = "VietNamese-llama-2-7b-chat-Medical"

def setup_cache_directory(cache_dir):
    """
    Creates the cache directory if it does not exist.

    Args:
        cache_dir (str): The path to the cache directory.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

def load_tokenizer_and_model(base_model, use_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, use_nested_quant):
    # Tạo compute_dtype từ tên kiểu dữ liệu
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Cấu hình BitsAndBytesConfig cho quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return tokenizer, model

def setup_peft_config(alpha, dropout, r):
    """
    Sets up the PeftConfig for the model.

    Args:
        alpha (int): The alpha value for LoRA.
        dropout (float): The dropout rate.
        r (int): The r value for LoRA.

    Returns:
        peft_config: The PeftConfig object.
    """
   # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft_config

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: The model object.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_training_dataset(file_path):
    return pd.read_csv(file_path)

def login_to_huggingface(token):
    try:
        login(token=token)
        print("Login successful!")
    except Exception as e:
        print(f"Login failed: {e}")

def login_to_wandb(api_key):
    """
    Đăng nhập vào Weights & Biases (W&B).
    
    Nếu api_key không được cung cấp, W&B sẽ yêu cầu nhập API key trong terminal.
    
    Args:
        api_key (str): API key của Weights & Biases. Nếu không cung cấp, W&B sẽ yêu cầu nhập trong terminal.
    """
    if api_key:
        # Đăng nhập bằng API key nếu có
        wandb.login(key=api_key)
        print("Successfully logged in to Weights & Biases.")
    else:
        # Đăng nhập thông qua prompt trong terminal
        wandb.login()
        print("Login prompt for Weights & Biases displayed.")


def setup_training_arguments(output_dir, per_device_train_batch_size, gradient_accumulation_steps, 
                             optim, save_steps, logging_steps, learning_rate, max_grad_norm, 
                             max_steps, warmup_ratio, lr_scheduler_type):
    """
    Sets up the training arguments for the trainer.

    Args:
        output_dir (str): The output directory for saving checkpoints and logs.
        per_device_train_batch_size (int): The batch size per device for training.
        gradient_accumulation_steps (int): The number of steps to accumulate gradients.
        optim (str): The optimizer to use.
        save_steps (int): The number of steps between saving checkpoints.
        logging_steps (int): The number of steps between logging training metrics.
        learning_rate (float): The learning rate for the optimizer.
        max_grad_norm (float): The maximum gradient norm for gradient clipping.
        max_steps (int): The maximum number of training steps.
        warmup_ratio (float): The warmup ratio for learning rate scheduling.
        group_by_length (bool): Whether to group samples by length during training.
        lr_scheduler_type (str): The type of learning rate scheduler to use.
        use_wandb (bool): Whether to use wandb for logging.
        wandb_project (str): The wandb project name.
        wandb_run_name (str): The wandb run name.
        wandb_watch (str): The wandb watch mode.
        wandb_log_model (str): The wandb log model mode.

    Returns:
        training_arguments: The TrainingArguments object.
    """
    # # Nếu W&B được sử dụng, khởi tạo dự án và cấu hình Weights & Biases
    # if report_to_wandb:
    #     wandb.init(
    #         project=wandb_project,     # Tên dự án trên Weights & Biases
    #         name=wandb_run_name,       # Tên phiên chạy trên Weights & Biases
    #         config={
    #             "learning_rate": learning_rate,
    #             "batch_size": per_device_train_batch_size,
    #             "optimizer": optim,
    #             "lr_scheduler_type": lr_scheduler_type
    #         }
    #     )
    #     if wandb_log_model:
    #         wandb.log_artifact("model")  # Log mô hình nếu được yêu cầu
            
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        # report_to_wandb=True,  # Xác định có sử dụng W&B không
        # wandb_project="vietnamese LLMs",  # Tên project W&B
        # wandb_run_name="SFT_LLaMA_7B_QLORA_Medical_Vietnamese",
        # wandb_log_model=True,  # Đăng tải mô hình lên W&B nếu True
    )
    return training_arguments

def prepare_trainer(model, dataset, peft_config, tokenizer, max_seq_length, training_arguments):
    """
    Prepares the SFTTrainer for training.

    Args:
        model: The model object.
        dataset: The training dataset.
        peft_config: The PeftConfig object.
        tokenizer: The tokenizer object.
        max_seq_length (int): The maximum sequence length.
        training_arguments: The TrainingArguments object.

    Returns:
        trainer: The SFTTrainer object.
    """
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments
    )
    return trainer

def main():
    cache_dir = "/data/rick/pretrained_weights/LLaMA/"
    setup_cache_directory(cache_dir)
    login_to_huggingface('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')   
    # login_to_wandb('ece8092446612fcc1ba38496e835dd6cbeafd165')
    tokenizer, model = load_tokenizer_and_model(base_model, use_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, use_nested_quant)
    alpha = 16
    dropout = 0.1
    r = 64
    peft_config = setup_peft_config(alpha, dropout, r)
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    dataset=load_training_dataset("C:/Users/nguye/OneDrive/Desktop/firetunne-LLMs-/data/train.csv")

    output_dir = "./results"
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 4
    optim = "adamw_bnb_8bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 100
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"
    # wandb_project = "Vietnamese_LLMs"
    # wandb_run_name = "SFT_LLaMA_7B_QLORA_Medical_Vietnamese"
    # wandb_log_model = "true"

    # use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)

    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    training_arguments = setup_training_arguments(output_dir, per_device_train_batch_size,
                                                  gradient_accumulation_steps, optim, save_steps, logging_steps,
                                                  learning_rate, max_grad_norm, max_steps, warmup_ratio,
                                                  lr_scheduler_type)

    max_seq_length = 2048
    trainer = prepare_trainer(model, dataset, peft_config, tokenizer, max_seq_length, training_arguments)

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

if __name__ == "__main__":
    main()
