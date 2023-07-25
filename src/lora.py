import os
import torch
import torch.nn as nn
import transformers
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset
from huggingface_hub import login

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_model():
    model_name = "gpt2-small"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True, 
        device_map='auto',
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2-small")
    model = prep_model(model)
    return model, tokenizer


def prep_model(model): 
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    #cast language modeling head to float
    model.lm_head = CastOutputToFloat(model.lm_head)
    
    return model

def get_trainable_params(model):
    all_params = 0 ; trainable_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    print(
        f"Total params: {all_params:,}, Trainable params: {trainable_params:,} ({trainable_params/all_params:.2%})"
    )
        

def get_lora_model(model, r=16, lora_alpha=32):   
    config = LoraConfig(
        r=r, #attention heads
        lora_alpha=lora_alpha, #alpha scaling
        # target_modules=["q_proj", "v_proj"], #if you know the 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )

    model = get_peft_model(model, config)
    print("Set up Lora model...")
    get_trainable_params(model)
    
    
def get_dataset():
    dataset_name = "vekkt/french_CEFR"
    data = load_dataset(dataset_name)
    
    
    
    
def train(model, dataset, tokenizer,): 
    assert torch.cuda.is_available()
    
    trainer = transformers.Trainer(
            model=model, 
            train_dataset=dataset['train'],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=4, 
                gradient_accumulation_steps=4,
                warmup_steps=100, 
                max_steps=200, 
                learning_rate=2e-4, 
                fp16=True,
                logging_steps=1, 
                output_dir='outputs'
                ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
                )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    return model
    
    
    
def main():
    login()
    model, tokenizer = load_model()
    model = get_lora_model(model)
    dataset = get_dataset()
    model = train(model, dataset, tokenizer)
    
    
if __name__ == "__main__":
    main()
    
    
    
    