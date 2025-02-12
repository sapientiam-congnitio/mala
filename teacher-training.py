from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import json
import evaluate
import numpy as np
import torch

with open("/gpfs/helios/home/manuchek/mala/data/classes.json") as file:
    classes = json.load(file)["data"]

class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}

model_path = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_function(example):
    text = example["text"].replace("\n", " ")

    labels = torch.zeros(len(classes), dtype=torch.float)
    label_id = class2id[example["label"]]
    labels[label_id] = 1.0

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=256,
    )
    encoding["labels"] = labels

    return encoding


train_file = "/gpfs/helios/home/manuchek/mala/data/train_valid_test_data/train/data-00000-of-00001.arrow"
valid_file = "/gpfs/helios/home/manuchek/mala/data/train_valid_test_data/valid/data-00000-of-00001.arrow"


ds_train = Dataset.from_file(train_file)
ds_valid = Dataset.from_file(valid_file)

features = list(ds_train.features)

tokenized_train_dataset = ds_train.map(preprocess_function)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(features)

tokenized_valid_dataset = ds_valid.map(preprocess_function)
tokenized_valid_dataset = tokenized_valid_dataset.remove_columns(features)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

clf_metrics = evaluate.combine(["accuracy"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)

    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(
        predictions=predictions, references=labels.astype(int).reshape(-1)
    )


model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(classes),
    id2label=id2class,
    label2id=class2id,
    problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    output_dir="/gpfs/helios/home/manuchek/mala/data/teacher_models/20-epochs",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
