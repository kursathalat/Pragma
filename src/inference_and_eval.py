# This script can be used to evaluate the model performance on a test set.
# Note that we give a classification task to a causal language model, which requires a different evaluation script.
# Train and test sets should include proper prompts with ground labels.
# Make sure you have access to gated 

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


# load the model
model_id =  "path/to/your/model"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

def inference(text):
  """
  Generate completions for the given text using the model.

    Args:
        text (str): The text for which completions are to be generated.
  """

  inputs = tokenizer(text, return_tensors="pt").to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=30)

  return tokenizer.decode(outputs[0], skip_special_tokens=True)


labels = ["THE LABELS YOU WANT TO PREDICT"]


def get_predictions(dataset):
    """
    Generate predictions for the given dataset.

    Args:
        dataset (Dataset): The dataset for which predictions are to be generated.
                            Note that dataset should have a 'text' and 'label' column.
    """

    global labels
    response_list = []
    prediction_list = []
    completion_list = []
    current_index = 0
    for item in dataset:
        print(current_index)
        current_index += 1
        n_tries = 0
        completion = ""
        while not any(label in completion for label in labels) and n_tries < 6:
            response = inference(item["text"])
            completion = response[len(item["text"]):]
            n_tries += 1
            if n_tries > 1:
                print("TRYING AGAIN, NO LABEL FOUND")
        response_list.append(response)
        completion_list.append(completion)

        # in case you have 4 labels, if more, please add more elifs
        if labels[0] in completion.lower():
            prediction_list.append(labels[0])
        elif labels[1] in completion.lower():
            prediction_list.append(labels[1])
        elif labels[2] in completion.lower():
            prediction_list.append(labels[2])
        elif labels[3] in completion.lower():
            prediction_list.append(labels[3])
        else:
            prediction_list.append("NA")

    gold_labels = dataset["label"]
    results = pd.DataFrame({'responses': response_list, 'completion': completion_list, 'predicted_labels': prediction_list, 'gold_labels': gold_labels})
    return results

results = get_predictions(dataset)

## Metrics

set(results["predicted_labels"].unique()).issubset(set(results["gold_labels"].unique()))

# metrics

# to see if we have any other value
set(results["predicted_labels"].unique()).issubset(set(results["gold_labels"].unique()))

#if needed, ds with only valid labels
#results_valid = llama_responses_test[llama_responses_test["predicted_labels"].isin(llama_responses_test["gold_labels"])]

preds = list(results["predicted_labels"])
labels = list(results["gold_labels"])

def compute_metrics(preds, labels):
  """
  Compute the metrics for two lists of given labels.

  preds: list of predicted labels
  labels: list of gold labels
  """
  acc = accuracy_score(labels, preds)
  f1 = f1_score(labels, preds, average='weighted')
  precision = precision_score(labels, preds, average='weighted')
  recall = recall_score(labels, preds, average='weighted')

  return {"accuracy": acc, "f1_score": f1, "precision": precision, "recall": recall}

metrics = compute_metrics(preds, labels)
