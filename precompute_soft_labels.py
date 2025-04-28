import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datasets import Dataset, concatenate_datasets

with open("") as file:  # classes path
    classes = json.load(file)["data"]

class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}


train_file = ""  # training data path
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


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


def get_teacher_predictions(model, df, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(df, batch_size=batch_size)

    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):

            batch_ids = batch["input_ids"].squeeze(1).to(device)
            batch_mask = batch["attention_mask"].squeeze(1).to(device)

            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)

            # Apply temperature scaling and softmax
            temperature = 2.0  # Adjust temperature as needed
            logits = outputs.logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def main():
    model = AutoModelForSequenceClassification.from_pretrained("")  # teacher model path

    train_dataset = Dataset.from_file(train_file)

    train_dataset = train_dataset.map(preprocess_function)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    soft_labels = get_teacher_predictions(
        model,
        train_dataset,
    )

    train_dataset = train_dataset.add_column("soft_labels", soft_labels.tolist())
    train_dataset.save_to_disk("")  # dataset that has teacher labels


if __name__ == "__main__":
    main()
