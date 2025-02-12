from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import Dataset
from collections import Counter
import torch
import json
import time

benchmark_dataset_path = "/gpfs/helios/home/manuchek/mala/data/train_valid_test_data/test/data-00000-of-00001.arrow"
model_name = "microsoft/deberta-v3-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/gpfs/helios/home/manuchek/mala/data/classes.json") as file:
    classes = json.load(file)["data"]


class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}


def preprocess(example, tokenizer):
    text = example["text"]
    tokenized_text = tokenizer(text, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]

    chunks = [tokenized_text[i : i + 254] for i in range(0, len(tokenized_text), 254)]
    chunks_with_special_tokens = [
        [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id] for chunk in chunks
    ]

    padded_input_ids_list = [
        chunk + [tokenizer.pad_token_id] * (256 - len(chunk))
        for chunk in chunks_with_special_tokens
    ]

    attention_masks_list = [
        [1] * len(chunk) + [0] * (256 - len(chunk))
        for chunk in chunks_with_special_tokens
    ]

    input_ids_list = [
        torch.tensor(chunk, dtype=torch.long).to(device)
        for chunk in padded_input_ids_list
    ]
    attention_masks_list = [
        torch.tensor(mask, dtype=torch.long).to(device) for mask in attention_masks_list
    ]

    example["chunks"] = chunks_with_special_tokens
    example["input_ids"] = input_ids_list
    example["attention_masks"] = attention_masks_list

    return example


dataset = Dataset.from_file(benchmark_dataset_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_teacher = AutoModelForSequenceClassification.from_pretrained(
    "/gpfs/helios/home/manuchek/mala/data/teacher_models/20-epochs/checkpoint-72460"
)
model_student = AutoModelForSequenceClassification.from_pretrained(
    "/gpfs/helios/home/manuchek/mala/data/best_student"
)

model_teacher.to(device)
model_student.to(device)

dataset = dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer})


def inference(dataset, model, device):
    all_predictions = []
    correct = 0
    total = 0

    for example in dataset:
        true_label = class2id[example["label"]]
        predicted_chunks = []
        for input, attention in zip(example["input_ids"], example["attention_masks"]):

            with torch.no_grad():
                outputs = model(
                    torch.tensor(input).to(device).unsqueeze(0),
                    torch.tensor(attention).to(device).unsqueeze(0),
                )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            predicted_chunks.append(predicted_label)

        most_common = Counter(predicted_chunks).most_common(1)[0][0]
        if most_common == true_label:
            correct += 1
        total += 1

    accuracy = correct / total
    print("Total accuracy:", accuracy)


if __name__ == "__main__":
    start = time.time()
    inference(dataset, model_teacher, device)
    end = time.time()
    total = end - start
    print("Total teacher model run time:", total)

    start = time.time()
    inference(dataset, model_student, device)
    end = time.time()
    total = end - start
    print("Total student model run time:", total)
