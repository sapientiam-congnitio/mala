from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datasets import Dataset, concatenate_datasets

with open("") as file:  # classes path
    classes = json.load(file)["data"]

class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}


def preprocess_function(example, tokenizer):
    text = (
        " ".join(example["text"])
        if isinstance(example["text"], list)
        else example["text"]
    )

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
    encoding["input_ids"] = encoding["input_ids"]
    encoding["attention_mask"] = encoding["attention_mask"]

    return encoding


probs_0 = {}
num_0 = 0

probs_10 = {}
num_10 = 0

probs_20 = {}
num_20 = 0

epoch_num = 0

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


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Dict[str, float]:

    global probs_0, num_0, probs_10, num_10, probs_20, num_20

    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_kd_loss = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:

        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].squeeze(1).to(device)
        labels = batch["labels"].squeeze(1).to(device)
        soft_labels = batch["soft_labels"].squeeze(1).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        student_logits = outputs.logits
        student_probs = torch.nn.functional.log_softmax(
            student_logits / temperature, dim=-1
        )

        if epoch_num == 0 and num_0 == 0:
            probs_0[num_0] = [
                {
                    "teacher": soft_labels.cpu().detach().numpy().tolist(),
                    "student": student_probs.cpu().detach().numpy().tolist(),
                }
            ]
            num_0 += 1
        elif epoch_num == 9 and num_10 == 0:
            probs_10[num_10] = [
                {
                    "teacher": soft_labels.cpu().detach().numpy().tolist(),
                    "student": student_probs.cpu().detach().numpy().tolist(),
                }
            ]
            num_10 += 1
        elif epoch_num == 19 and num_20 == 0:
            probs_20[num_20] = [
                {
                    "teacher": soft_labels.cpu().detach().numpy().tolist(),
                    "student": student_probs.cpu().detach().numpy().tolist(),
                }
            ]
            num_20 += 1

        kd_loss = torch.nn.KLDivLoss(reduction="batchmean")(
            student_probs, soft_labels
        ) * (temperature**2)

        bce_loss = torch.nn.BCEWithLogitsLoss()(student_logits, labels)

        loss = (alpha * kd_loss) + ((1 - alpha) * bce_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += bce_loss.item()
        total_kd_loss += kd_loss.item()

        progress_bar.set_postfix(
            {
                "loss": loss.item(),
                "ce_loss": bce_loss.item(),
                "kd_loss": kd_loss.item(),
            }
        )

    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_kd_loss = total_kd_loss / num_batches

    return {"loss": avg_loss, "ce_loss": avg_ce_loss, "kd_loss": avg_kd_loss}


def validate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].squeeze(1).to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            labels = batch["labels"].squeeze(1).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels.float())

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)

            total_correct += (predictions == torch.argmax(labels, dim=-1)).sum().item()

            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    return {"val_loss": avg_loss, "val_accuracy": accuracy}


best_val_loss = float("inf")
best_val_accuracy = float("-inf")


def main():
    global best_val_loss, epoch_num

    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    train_file_1 = ""
    train_file_2 = ""
    train_file_3 = ""
    train_file_4 = ""
    train_file_5 = ""
    valid_file = ""

    train_dataset = concatenate_datasets(
        [
            Dataset.from_file(train_file_1),
            Dataset.from_file(train_file_2),
            Dataset.from_file(train_file_3),
            Dataset.from_file(train_file_4),
            Dataset.from_file(train_file_5),
        ]
    )
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "soft_labels"],
    )

    valid_dataset = Dataset.from_file(valid_file)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    valid_dataset = valid_dataset.map(preprocess_function)
    valid_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)

    model_name = ""  # student model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    num_epochs = 20
    temperature = 2.0
    alpha = 0.1

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, temperature, alpha
        )
        print(f"Average loss: {train_metrics['loss']:.4f}")
        print(f"Average CE loss: {train_metrics['ce_loss']:.4f}")
        print(f"Average KD loss: {train_metrics['kd_loss']:.4f}")

        val_metrics = validate(model, val_loader, device)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            model.save_pretrained("")  # save best student model

            print(f"Saved new best model with loss: {best_val_loss:.4f}")
            print(f"Accuracy: {val_metrics['val_accuracy']:.4f}")

        epoch_num += 1


if __name__ == "__main__":
    main()
