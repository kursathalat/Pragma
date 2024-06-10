# Certain versions of transformers are not compatible with other packages, so it should be uninstalled first

#!pip uninstall transformers -y
#!pip install accelerate -q
#!pip install transformers==4.30.0 -q
#!pip install -q -U peft datasets trl bitsandbytes -q

# Importing necessary libraries
from datasets import Dataset
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import re
from peft import LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import os
from trl import SFTTrainer

#from huggingface_hub import notebook_login # for gated access to private models
#from google.colab import drive #for those working on google colab

# drive.mount('/content/gdrive') #suggested for saving the model
# notebook_login() #for gated access to private models

model_id =  "google/gemma-7b-it"

bnb_config = BitsAndBytesConfig( #quantize the model to 4-bit for faster inference (but not with that much decrease in accuracy)
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

lora_config = LoraConfig(
    r=8, #rank should be 8
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


ds_train_pd = pd.read_excel("train_dataset.xlsx") #first upload data from any source like xlsx, csv, etc.
ds_train = Dataset.from_pandas(ds_train_pd) #convert to dataset
#ds_train = ds_train.shard(num_shards=130, index=6) #for experiments with different shards


# Use SFTTrainer from trl, as it is a subclass of transformers.Trainer

trainer = SFTTrainer(
    model=model,
    train_dataset=ds_train,

    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
)
trainer.train()

trainer.save_model("path/to/save/model") #save the model to a path

# Note that only the adapters will be saved in the model, not the full model.