from copy import deepcopy
from datasets import load_dataset
from hedgehog_roberta import RobertaHedgehogSelfAttention
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention
import evaluate
import numpy as np
import os
import torch
import transformers


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


def replace_layer(module):
    if isinstance(module, RobertaSelfAttention):
        # target_state_dict = deepcopy(module.state_dict())
        new_module = RobertaHedgehogSelfAttention(module)
        # new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN
    )


MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-05

os.environ["WANDB_PROJECT"] = "hedgehog_roberta_twitter_sentiment"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=3, output_attentions=False
)


for name, module in tuple(model.named_modules()):
    if name:
        recursive_setattr(model, name, replace_layer(module))

model = model.to(torch.bfloat16)

dataset = load_dataset("tweet_eval", "sentiment")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

metric = evaluate.load("accuracy")


training_args = TrainingArguments(
    bf16=True,
    dataloader_num_workers=8,
    do_eval=True,
    evaluation_strategy="steps",
    gradient_checkpointing=True,
    learning_rate=LR,
    load_best_model_at_end=True,
    logging_dir="logging",
    metric_for_best_model="accuracy",
    num_train_epochs=EPOCHS,
    optim="adamw_torch_fused",
    output_dir="output",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    report_to=["wandb"],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    args=training_args,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    eval_dataset=tokenized_dataset["validation"],
    model=model,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

trainer.create_model_card()
trainer.save_model("saved_model")
