import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import time
from datasets import load_dataset
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()

set_seed(42)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('andstor/gpt-j-6B-smart-contract', use_auth_token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained('andstor/gpt-j-6B-smart-contract', use_auth_token=os.environ['HF_TOKEN'], pad_token_id=tokenizer.eos_token_id).to(device)

    datasets = load_dataset("andstor/smart_contract_code_comments", use_auth_token=os.environ['HF_TOKEN'])
    shuffled_ds = datasets.shuffle(seed=42)
    dataset = shuffled_ds["test"]

    #----------------------------------------------------------------------------------------------------------------------

    from random import randint
    from transformers import StoppingCriteriaList
    from src.generation.matching_braces_criteria import MatchingBracesCriteria
    from src.evaluation.bleu_score import calc_bleu_score


    def generate(contract, slack=256, context=False):

        if context:
            start_index = contract["meta"]["func_code_index"][0]
            end_index = contract["meta"]["func_code_index"][1]
            target_func_code = contract["class_code"][start_index:end_index+1] # Without adjusted indentations
            text = contract["class_code"][:end_index+1]
            target_gen_length = len(tokenizer(target_func_code).input_ids)
        else:
            text = contract["func_documentation"]
            target_func_code = contract["func_code"] # With adjusted indentations
            target_gen_length = len(tokenizer(contract["func_code"]).input_ids)
        
        gen_length = target_gen_length + slack
        gen_length = min(gen_length, model.config.max_position_embeddings - slack)

        # ---------------------------------------------
        # Truncuate if input_ids length > max_position_embeddings
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_length = len(input_ids[0])
        truncated = None
        if (input_length) > (model.config.max_position_embeddings - gen_length):
            print("Truncating!")
            truncated = True
            input_ids = input_ids[:, input_length - (model.config.max_position_embeddings - gen_length):]
        else:
            truncated = False
        input_ids = input_ids.to(device)

        # ---------------------------------------------
        # Print stats
        input_length = len(input_ids[0])
        print("input_length: " + str(input_length))
        print("target_funcion_length: " + str(target_gen_length))
        print("generation_length: " + str(gen_length))

        # ---------------------------------------------
        # Generate code
        text = "" # Only used if starting in the middle of a function

        # generate text until the output length (which includes the context length) reaches 50
        with torch.no_grad():
            greedy_output = model.generate(
                input_ids,
                stopping_criteria=StoppingCriteriaList([MatchingBracesCriteria(tokenizer, text)]),
                do_sample=False,
                max_new_tokens=gen_length
                #output_scores=True,
                #return_dict_in_generate=True
            )

        original_text = tokenizer.decode(input_ids[0])
        output_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        new_text = tokenizer.decode(greedy_output[0][input_ids.shape[1]:])
        target_function = contract["func_code"]
        
        bleu_score = calc_bleu_score(target_function, new_text)
        return {
            "input": original_text,
            "output": output_text,
            "generated": new_text,
            "target": target_function,
            "bleu_score": bleu_score,
            "truncated": truncated,
        }


    #----------------------------------------------------------------------------------------------------------------------



    ds = dataset#dataset.filter(lambda example: example["label"] == 2)

    data = []

    for i in range(args.n_samples):
        print("Iteration: " + str(i) + "/" + str(args.n_samples))
        contract = ds[i]
        start_time = time.time()

        result = generate(contract, slack=args.slack, context=args.context)

        end_time = time.time()
        duration = end_time - start_time

        data.append({
            **contract,
            "input": result['input'],
            "output": result['output'],
            "generated": result['generated'],
            "target": result['target'],
            "truncated": result["truncated"],
            "duration": duration,
            "bleu_score": result["bleu_score"],
        })

        print("Bleu score: " + str(result["bleu_score"]))
        print("--- %s seconds ---" % (duration))

    df_res = pd.DataFrame(data)
    
    df_res.to_parquet("generated_comments_slack_" + str(args.slack) + ".parquet")

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--slack", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()

    main(args)

