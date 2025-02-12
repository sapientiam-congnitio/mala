import re

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import torch


def extract_layer_num(key):
    patterns = [
        r"\.layer\.(\d+)\.",  # BERT style
        r"encoder\.layer\.(\d+)\.",  # BERT encoder style
        r"decoder\.layer\.(\d+)\.",  # Decoder style
        r"\.layers\.(\d+)\.",  # GPT style
        r"\.block\.(\d+)\.",  # ViT style
        r"h\.(\d+)\.",  # GPT-2 style
    ]

    for pattern in patterns:
        match = re.search(pattern, key)
        if match:
            return int(match.group(1))
    return None


def get_layer_prefix(key):
    patterns = {
        r"encoder\.layer\.\d+\.": "encoder.layer.",
        r"decoder\.layer\.\d+\.": "decoder.layer.",
        r"layer\.\d+\.": "layer.",
        r"layers\.\d+\.": "layers.",
        r"block\.\d+\.": "block.",
        r"h\.\d+\.": "h.",
    }

    for pattern, prefix in patterns.items():
        if re.search(pattern, key):
            return prefix
    return None


def verify_layer_equality(original_model, stripped_model, layers_to_remove):
    """
    Verify that the remaining layers in the stripped model are identical to the original.
    """
    original_dict = original_model.state_dict()
    stripped_dict = stripped_model.state_dict()

    comparison_results = {
        "identical_layers": [],
        "mismatched_layers": [],
        "weight_differences": {},
    }

    layer_nums = set()
    for key in original_dict.keys():
        layer_num = extract_layer_num(key)
        if layer_num is not None:
            layer_nums.add(layer_num)

    total_layers = max(layer_nums) + 1
    remaining_layers = [i for i in range(total_layers) if i not in layers_to_remove]
    layer_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_layers)}

    for key in original_dict.keys():
        print(f"Current key: {key}")
        layer_num = extract_layer_num(key)
        if layer_num is not None and layer_num not in layers_to_remove:
            # Get the layer prefix for this architecture
            prefix = get_layer_prefix(key)
            if prefix is None:
                continue

            # Map to new layer number
            new_layer_num = layer_map[layer_num]
            new_key = re.sub(f"{prefix}{layer_num}", f"{prefix}{new_layer_num}", key)
            print("dlsjfaldks", new_layer_num)

            # Compare weights
            if new_key in stripped_dict:
                original_weights = original_dict[key]
                stripped_weights = stripped_dict[new_key]

                if torch.equal(original_weights, stripped_weights):
                    comparison_results["identical_layers"].append(layer_num)
                else:
                    comparison_results["mismatched_layers"].append(layer_num)
                    diff = (original_weights - stripped_weights).abs()
                    comparison_results["weight_differences"][layer_num] = {
                        "max_diff": float(diff.max()),
                        "mean_diff": float(diff.mean()),
                        "std_diff": float(diff.std()),
                    }

    return len(comparison_results["mismatched_layers"]) == 0, comparison_results


def strip_transformer_layers(
    model_name,
    layers_to_remove=None,
    keep_first_n=None,
    output_dir=None,
    num_labels=None,
    verify=True,
):
    if layers_to_remove is None and keep_first_n is None:
        raise ValueError("Must specify either layers_to_remove or keep_first_n")

    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name)

    # Set number of labels if specified
    if num_labels is not None:
        config.num_labels = num_labels

    # Determine total number of layers
    if hasattr(config, "num_hidden_layers"):
        total_layers = config.num_hidden_layers
        layer_attr = "num_hidden_layers"
    elif hasattr(config, "num_layers"):
        total_layers = config.num_layers
        layer_attr = "num_layers"
    else:
        raise ValueError("Model architecture not supported for layer stripping")

    print(f"Total layers: {total_layers}")

    if keep_first_n is not None:
        if keep_first_n >= total_layers:
            return AutoModelForSequenceClassification.from_pretrained(model_name), None
        layers_to_remove = list(range(keep_first_n, total_layers))

    print("Layers to remove", layers_to_remove)
    if max(layers_to_remove) >= total_layers or min(layers_to_remove) < 0:
        raise ValueError(f"Layer indices must be between 0 and {total_layers-1}")

    remaining_layers = [i for i in range(total_layers) if i not in layers_to_remove]
    layer_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_layers)}

    print(layer_map)

    # Update config with new number of layers
    setattr(config, layer_attr, len(remaining_layers))

    original_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    stripped_model = AutoModelForSequenceClassification.from_config(config)
    print(stripped_model)

    stripped_dict = stripped_model.state_dict()
    original_dict = original_model.state_dict()

    for key in stripped_dict.keys():
        layer_num = extract_layer_num(key)
        if layer_num is not None:
            for orig_key in original_dict.keys():
                orig_layer_num = extract_layer_num(orig_key)
                if (
                    orig_layer_num is not None
                    and orig_layer_num not in layers_to_remove
                    and layer_map[orig_layer_num] == layer_num
                    and key.replace(str(layer_num), str(orig_layer_num)) == orig_key
                ):

                    stripped_dict[key] = original_dict[orig_key]
                    break
        else:
            if key in original_dict:
                stripped_dict[key] = original_dict[key]

    print(stripped_dict.keys())

    # # Load the modified state dict
    stripped_model.load_state_dict(stripped_dict)
    # Verify layer equality if requested
    verification_results = None
    if verify:
        is_identical, verification_results = verify_layer_equality(
            original_model, stripped_model, layers_to_remove
        )
        if not is_identical:
            print("Warning: Some layers do not match exactly!")
            print("Mismatched layers:", verification_results["mismatched_layers"])
            print("Weight differences:", verification_results["weight_differences"])

    # Save the model if output directory is provided
    if output_dir:
        stripped_model.save_pretrained(output_dir)
        config.save_pretrained(output_dir)

    return stripped_model, verification_results


if __name__ == "__main__":
    models_to_test = [
        "/gpfs/helios/home/manuchek/mala/data/teacher_models/20-epochs/checkpoint-72460"
    ]
    # models_to_test = ["microsoft/deberta-v3-small"]

    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        try:
            stripped_model, verification_results = strip_transformer_layers(
                model_name=model_name, keep_first_n=2, verify=True, num_labels=424
            )
            print("Successfully stripped layers")
            if verification_results:
                print(f"Identical layers: {verification_results['identical_layers']}")
                print(f"Mismatched layers: {verification_results['mismatched_layers']}")
        except Exception as e:
            print(f"Failed: {str(e)}")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

        premise = "A few people in a restaurant setting, one of them is drinking orange juice."
        hypothesis = "The diners are at a restaurant."
        label_names = ["entailment", "neutral", "contradiction"]
        inputs = tokenizer(
            premise, hypothesis, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = stripped_model(**inputs)
            stripped_model_predictions = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            ).tolist()[0]

        stripped_model_predictions = {
            name: round(float(pred) * 100, 1)
            for pred, name in zip(stripped_model_predictions, label_names)
        }
        print(stripped_model_predictions)

        original_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        with torch.no_grad():
            outputs = original_model(**inputs)
            original_model_predictions = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            ).tolist()[0]

        original_model_predictions = {
            name: round(float(pred) * 100, 1)
            for pred, name in zip(original_model_predictions, label_names)
        }
        print(original_model_predictions)

        print(original_model)
        stripped_model.save_pretrained(
            "/gpfs/helios/home/manuchek/mala/data/student_models/stripped-20-epochs"
        )
