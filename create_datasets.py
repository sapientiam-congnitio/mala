from datasets import Dataset, concatenate_datasets
import os
import json
import random
from typing import Tuple, Union, List


main_path = "mala/languages/"
languages = os.listdir(main_path)
lang_paths = {
    language: os.path.join(main_path, language)
    for language in languages
    if language != ".cache"
}

all_train_sets = []
all_valid_sets = []
all_test_sets = []


def add_label_column(ds: Dataset, lang: str):
    return ds.add_column("label", [lang] * len(ds))


def split_text(
    original_text: str,
    train_valid_ratio: float = 0.8,
    train_ratio: float = 0.8,
    include_test: bool = True,
) -> Tuple[str, str, str]:

    tv_len = int(len(original_text) * train_valid_ratio)
    train_split = int(tv_len * train_ratio)

    train_text = original_text[:train_split]
    valid_text = original_text[train_split:tv_len]
    test_text = original_text[tv_len:] if include_test else ""

    if len(train_text) > 3:
        split1 = int(len(train_text) / 3)
        part1 = train_text[:split1]
        part2 = train_text[split1 : split1 * 2]
        part3 = train_text[split1 * 2 :]

        train_texts = [part1, part2, part3]
    elif len(train_text) > 2:
        train_texts = [part1, part2, part3]
        split1 = int(len(train_text) / 2)
        part1 = train_text[:split1]
        part2 = train_text[split1:]
        train_texts = [part1, part2]
    else:
        train_texts = [train_text]

    return train_texts, valid_text, test_text


def create_ds(base_ds: Dataset, texts: Union[List[str], str]) -> Dataset:
    columns = base_ds.column_names

    if isinstance(texts, list):
        new_ds = Dataset.from_dict(
            {col: [base_ds[col][0]] * len(texts) for col in columns}
        )
        new_ds = new_ds.map(lambda x, i: {"text": texts[i]}, with_indices=True)

    else:
        new_ds = Dataset.from_dict({col: [base_ds[col][0]] for col in columns})
        new_ds = new_ds.map(lambda x: {"text": texts})

    return new_ds


def process_full_dataset(
    ds_shuffled: Dataset, lang: str
) -> Tuple[Dataset, Dataset, Dataset]:

    test_ds = ds_shuffled.select(range(2))
    valid_ds = ds_shuffled.select(range(2, 202))
    train_ds = ds_shuffled.select(range(202, 1002))

    return train_ds, valid_ds, test_ds


def process_small_size_dataset(
    ds_shuffled: Dataset, lang: str
) -> Tuple[Dataset, Dataset, Dataset]:
    if len(ds_shuffled) == 1:
        original_text = ds_shuffled["text"][0]

        train_texts, valid_text, test_text = split_text(
            original_text, train_valid_ratio=0.8, train_ratio=0.8, include_test=True
        )

        train_ds = create_ds(ds_shuffled, train_texts)
        valid_ds = create_ds(ds_shuffled, valid_text)
        test_ds = create_ds(ds_shuffled, test_text)

    elif len(ds_shuffled) == 2:
        first_text = ds_shuffled["text"][0]
        second_text = ds_shuffled["text"][1]

        if len(first_text) < len(second_text):
            test_ds = ds_shuffled.select([0])
            train_valid_ds = ds_shuffled.select([1])
        else:
            test_ds = ds_shuffled.select([1])
            train_valid_ds = ds_shuffled.select([0])

        original_text = train_valid_ds["text"][0]

        train_texts, valid_text, test_text = split_text(
            original_text, train_valid_ratio=0.8, train_ratio=0.8, include_test=False
        )

        train_ds = create_ds(ds_shuffled, train_texts)
        valid_ds = create_ds(ds_shuffled, valid_text)

    return train_ds, valid_ds, test_ds


def process_exception_dataset(
    ds_shuffled: Dataset, lang: str
) -> Tuple[Dataset, Dataset, Dataset]:
    if len(ds_shuffled) == 1 or len(ds_shuffled) == 2:
        train_ds, valid_ds, test_ds = process_small_size_dataset(ds_shuffled, lang)

    else:

        train_valid_split, test_ds = ds_shuffled.train_test_split(
            test_size=0.2, seed=42
        ).values()
        train_ds, valid_ds = train_valid_split.train_test_split(
            test_size=0.2, seed=42
        ).values()

        if len(test_ds) > 2:
            test_ds_temp = test_ds

            test_ds = test_ds_temp.select(range(2))
            train_ds = concatenate_datasets(
                [train_ds, test_ds_temp.select(range(2, len(test_ds_temp)))]
            )

    return train_ds, valid_ds, test_ds


def main():
    for lang, path in lang_paths.items():
        for file_name in os.listdir(path):
            if file_name.startswith("data"):
                ds_shuffled = Dataset.from_file(os.path.join(path, file_name)).shuffle(
                    seed=42
                )
                if len(ds_shuffled) < 1002:
                    train_ds, valid_ds, test_ds = process_exception_dataset(
                        ds_shuffled, lang
                    )
                else:
                    train_ds, valid_ds, test_ds = process_full_dataset(
                        ds_shuffled, lang
                    )

                train_ds = add_label_column(train_ds, lang)
                valid_ds = add_label_column(valid_ds, lang)
                test_ds = add_label_column(test_ds, lang)

                all_train_sets.append(train_ds)

                all_valid_sets.append(valid_ds)

                all_test_sets.append(test_ds)

                print(
                    f"Processed {lang}: Train set size = {len(train_ds)}, Valid set size = {len(valid_ds)}, Test set size = {len(test_ds)} "
                )

    combined_train_set = concatenate_datasets(all_train_sets)
    combined_valid_set = concatenate_datasets(all_valid_sets)
    combined_test_set = concatenate_datasets(all_test_sets)
    print(len(combined_train_set), len(combined_valid_set), len(combined_test_set))

    combined_train_set.save_to_disk("mala/train_valid_test_data/train")
    combined_valid_set.save_to_disk("mala/train_valid_test_data/valid")
    combined_test_set.save_to_disk("mala/train_valid_test_data/test")


main()
