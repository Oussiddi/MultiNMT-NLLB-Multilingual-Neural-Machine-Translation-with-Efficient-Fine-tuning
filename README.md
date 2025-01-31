# Multilingual Neural Machine Translation with NLLB-200

## Project Overview
This project implements a multilingual neural machine translation system based on Meta's NLLB-200 model. It supports translation from French to multiple target languages including English, Spanish, Arabic, Chinese, and Russian, using efficient fine-tuning techniques like LoRA and mixed precision training.

## Supported Languages
- French (Source) → English
- French → Spanish
- French → Arabic
- French → Chinese (Simplified)
- French → Russian

## Model Architecture
- Base Model: `facebook/nllb-200-distilled-600M`
- Optimization: LoRA (Low-Rank Adaptation)
- Training Mode: Mixed Precision (FP16)
- Device Mapping: Automatic

## Features
- Efficient fine-tuning using LoRA
- Cosine learning rate scheduling with warmup
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- Frozen encoder for transfer learning
- Gradient accumulation

## Configuration
```python
MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 500
```

## LoRA Configuration
```python
LORA_CONFIG = {
    "r": 16,               # Rank
    "lora_alpha": 32,      # Scale
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "SEQ_2_SEQ_LM"
}
```

## Language Codes
```python
LANG_CODES = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'ar': 'ara_Arab',
    'zh': 'zho_Hans',
    'ru': 'rus_Cyrl'
}
```

## Training Features
1. Learning Rate Schedule
   - Warmup phase for stable training
   - Cosine decay for better convergence
   - Dynamic adjustment based on steps

2. Optimization
   - AdamW optimizer
   - Gradient accumulation (4 steps)
   - Mixed precision training
   - Gradient checkpointing

3. Memory Management
   - Frozen encoder layers
   - Efficient LoRA adaptation
   - Automatic device mapping

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
```python
# Prepare your dataset
dataset = prepare_dataset(your_dataframe)
```

### Training
```python
# Initialize model
model = init_model()

# Start training
train(model, train_dataset, tokenizer)
```

## Dataset Format
Input DataFrame should contain columns:
- 'fr' (French source text)
- 'en' (English translations)
- 'es' (Spanish translations)
- 'ar' (Arabic translations)
- 'zh' (Chinese translations)
- 'ru' (Russian translations)

## Monitoring
Training progress is logged every 100 steps:
- Loss values
- Current learning rate
- Batch progress
- Epoch progress

## Model Saving
- Checkpoints saved every 1000 steps
- Output directory: "./results"
- Logs directory: "./logs"

## Requirements
- transformers
- torch
- datasets
- peft
- accelerate
- bitsandbytes
- pandas
- numpy
