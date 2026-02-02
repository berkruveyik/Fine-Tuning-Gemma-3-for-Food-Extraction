# Fine-tune a Small LLM for Food Extraction

This project demonstrates supervised fine-tuning (SFT) of a small LLM to extract food and drink items from text. The workflow uses Hugging Face `transformers`, `trl`, and `datasets` to fine-tune **Gemma 3 270M** on a curated dataset of food-related captions.

## What It Does

- Loads `google/gemma-3-270m-it`
- Uses the dataset `mrdbourke/FoodExtract-1k`
- Converts samples into chat-style `messages` for SFT
- Trains with `TRL`'s `SFTTrainer`
- Evaluates by generating outputs on held-out samples
- Saves checkpoints and reloads a trained model for inference

## Project Structure

- `finetune_llm.ipynb`: Main notebook with the full training pipeline
- `checkpoint_models/`: Training outputs/checkpoints
- `llm_demo.ipynb` and `demo/`: (if used) demo assets
- `readme.md`: Project documentation

## Requirements

- Python 3.10+
- PyTorch
- `transformers`, `trl`, `datasets`, `huggingface_hub`, `matplotlib`
- GPU recommended (but not required)

## Setup

1. Install dependencies (example):

```bash
pip install torch transformers trl datasets huggingface_hub matplotlib
```


## Running the Notebook

Open and run `finetune_llm.ipynb` step by step. The notebook covers:

1. Model loading
2. Dataset download and inspection
3. Dataset formatting for chat-style SFT
4. Baseline generation
5. Prompting attempt without fine-tuning
6. Fine-tuning with `SFTTrainer`
7. Evaluation on test samples
8. Reloading a checkpoint for inference

## Training Configuration (Highlights)

- `num_train_epochs = 3`
- `per_device_train_batch_size = 8`
- `learning_rate = 5e-5`
- `max_length = 512`
- Checkpoints saved to `./checkpoint_models`

## Example Task

Input text:

```
I had a sandwich and a coffee for lunch.
```

Expected output format:

```
food_items = ['sandwich']
drink_items = ['coffee']
```

## Notes

- The dataset is converted into `messages` with `user` and `assistant` roles before training.
- The notebook shows how prompting alone fails for this specific extraction task, motivating fine-tuning.
- Checkpoints are saved locally and can be loaded with `AutoModelForCausalLM.from_pretrained`.

## Disclaimer

This is an educational demo. Results depend on compute, hyperparameters, and dataset quality.
