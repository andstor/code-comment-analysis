
from transformers import AutoTokenizer, AutoModel, set_seed, AutoConfig
from accelerate import Accelerator

import pandas as pd
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import json

from dotenv import load_dotenv
import os

# logger
import logging
logger = logging.getLogger(__name__)

load_dotenv()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Do inference with a transformer model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library). Can also a be \"csv\", \"json\", \"parquet\" or \"arrow\" for loading a local or remote file.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs='*',
        default=None,
        help="A list of data files to use (a local dataset path or a url).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The name of the split to use. Default to \"train\".",
    )
    parser.add_argument(
        "--text_column_names",
        type=str,
        nargs='*',
        default=None,
        help="The column names of the dataset to generate from.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--generation_config_file",
        type=str,
        default=None,
        help="Generation config path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="The output jsonline file where the data will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible generation."
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need a dataset name.")
    
    return args


def main():
    """
    Generate new data by sampling from the original data.
    """
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    print("omg")
    print(f'Num Processes: {accelerator.num_processes}; Device: {accelerator.device}; Process Index: {accelerator.process_index}')

    print(accelerator.device)


    if args.seed is not None:
        set_seed(args.seed)
    
    # Write the generation config to disk
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        token=os.environ['HF_TOKEN'],
        device_map="auto"
    )
    #model.eval() # not needed as from_pretrained() calls it already

    dataset = load_dataset(path=args.dataset_name, name=args.dataset_config_name, data_files=args.data_files, split=args.dataset_split)

    print("Dataset loaded!")
    column_names = dataset.column_names
    if args.text_column_names is not None:
        text_column_names = args.text_column_names
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]
        text_column_names = [text_column_name]
        logger.warning(f"Using column {text_column_name} as text column.")


    def get_embedding(input_text, tokenizer):
        #input_text = input_text.strip() # strip if you want to remove leading and trailing spaces
        encoded_input = tokenizer(input_text, return_tensors='pt')
        encoded_input.to(accelerator.device)
        output = model(**encoded_input)
        
        token_reps = output.last_hidden_state.squeeze(0)

        t_age_reps = torch.mean(token_reps, 0)
        t_max_reps = torch.max(token_reps, 0)
        t_max_reps = t_max_reps.values

        t_age_reps = t_age_reps.detach().cpu().numpy()
        t_max_reps = t_max_reps.detach().cpu().numpy()
        #t_ave_reps = ','.join(map(str, t_ave_reps))
        #t_max_reps = ','.join(map(str, t_max_reps))
        return t_age_reps, t_max_reps


    for row in tqdm(dataset, total=dataset.shape[0]):
        #g_code, o_code = row["generated"], row["func_code"]
        #if not g_code.strip() or not o_code.strip():
            #print("Empty code detected!")
            #continue

        for column in text_column_names:
            try:
                text = row[column]
                # get code embeddings
                avg_reps, max_reps = get_embedding(text, tokenizer)
                row[column + '_pooled_avg'] = avg_reps.tolist()
                row[column + '_pooled_max'] = max_reps.tolist()
            except Exception as e:
                print(e)
                raise e

        with open(args.output_file, 'a') as outfile:
            outfile.write(json.dumps(row))
            outfile.write('\n')


if __name__ == '__main__':
    main()
