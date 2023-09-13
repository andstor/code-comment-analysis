# code-comment-analysis
> Analysis of code comments using language models

## Description
This repository contains code for the analysis of code comments using language models. It provides necessary scripts to generate embeddings and to perform clustering analysis on the generated embeddings.

## Requirements

### Dependencies
Install the Python dependencies defined in the requirements.txt.
```bash
pip install -r requirements.txt
```

### Accelerate
Setup accelerate:
```bash
accelerate config
```

## Usage

### Generate embeddings

```bash
usage: generate-embeddings.py [-h] [--dataset_name DATASET_NAME] [--dataset_config_name DATASET_CONFIG_NAME] [--data_files [DATA_FILES ...]]
                              [--dataset_split DATASET_SPLIT] [--text_column_names [TEXT_COLUMN_NAMES ...]] [--model_name_or_path MODEL_NAME_OR_PATH]
                              [--config_name CONFIG_NAME] [--generation_config_file GENERATION_CONFIG_FILE] [--tokenizer_name TOKENIZER_NAME] [--use_slow_tokenizer]
                              [--output_file OUTPUT_FILE] [--seed SEED]

Do inference with a transformer model on a causal language modeling task

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library). Can also a be "csv", "json", "parquet" or "arrow" for loading a local or remote
                        file.
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the datasets library).
  --data_files [DATA_FILES ...]
                        A list of data files to use (a local dataset path or a url).
  --dataset_split DATASET_SPLIT
                        The name of the split to use. Default to "train".
  --text_column_names [TEXT_COLUMN_NAMES ...]
                        The column names of the dataset to generate from.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models.
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name
  --generation_config_file GENERATION_CONFIG_FILE
                        Generation config path if not the same as model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name
  --use_slow_tokenizer  If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).
  --output_file OUTPUT_FILE
                        The output jsonline file where the data will be saved.
  --seed SEED           A seed for reproducible generation.
```


#### Example

The following example will load the data from the parquet and generate embeddings. The output will be saved to the specified jsonl file.

```bash
accelerate launch script/generate-embeddings.py \
--dataset_name parquet \
--data_files generated_comments_slack_256.parquet \
--text_column_names generated func_code \
--model_name_or_path andstor/gpt-j-6B-smart-contract \
--output_file embeddings_generated_comments_slack_256.jsonl \
--seed 42
```

