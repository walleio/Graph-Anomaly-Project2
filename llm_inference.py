from openai import OpenAI
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from self_parser import parser
from utility import (transform_dict,  cs_categories, product_categories_raw, arxiv24_categories_raw, products_mapping,
                     save_dict_as_pickle, amazon_mapping)
import random
from self_parser import parser, args
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import torch

cs_categories_short = transform_dict(original_dict=cs_categories)
def gen_arxiv_message_original(context, version='zero-shot', example_input=None, example_output=None):
    sys_prompt = f"You are an AI trained to categorize arXiv computer science papers into specific categories based on their abstracts. Your task is to analyze the paper description provided and identify the most relevant category."
    user_prompt = f"Paper description: {context.strip()}\nGive me the category of this content. Respond only with the category key (e.g., 'cs.AI', 'cs.SY'), without any additional text or explanation."
    if version == 'zero-shot':
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}]
    if version == 'ex':
        if len(example_input) == 3:
            icl_str = f"Here are some of the papers cited by this paper: " \
                      f"\nPaper 1: \nContent: {example_input[0]}" \
                      f"\nPaper 2: \nContent: {example_input[1]}" \
                      f"\nPaper 3: \nContent: {example_input[2]}"
            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]

        elif len(example_input) == 2:
            icl_str = f"Here are some of the papers cited by this paper: " \
                      f"\nPaper 1: \nContent: {example_input[0]}" \
                      f"\nPaper 2: \nContent: {example_input[1]}"
            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]
        elif len(example_input) == 1:
            icl_str = f"Here are some of the papers cited by this paper: " \
                      f"\nPaper 1: \nContent: {example_input[0]}"
            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]
    elif version == 'icl':
        if len(example_input) == 3:
            icl_str = f"Here are some example: " \
                      f"\nExample 1: \nInput: {example_input[0]}\nOutput: {example_output[0]}\n\n" \
                      f"\nExample 2: \nInput: {example_input[1]}\nOutput: {example_output[1]}\n\n" \
                      f"\nExample 3: \nInput: {example_input[2]}\nOutput: {example_output[2]}"
            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]

        elif len(example_input) == 2:
            icl_str = f"Here are some example: " \
                      f"\nExample 1: \nInput: {example_input[0]}\nOutput: {example_output[0]}\n\n" \
                      f"\nExample 2: \nInput: {example_input[1]}\nOutput: {example_output[1]}"
            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]
        elif len(example_input) == 1:
            icl_str = f"This is an example: " \
                      f"\nExample 1: \nInput: {example_input[0]}\nOutput: {example_output[0]}"

            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]
        else:
            icl_str = f"Here are some example: "
            idx = 0
            for input, output in zip(example_input, example_output):

                trimmed_input = input[:900] if len(input) > 900 else input
                icl_str = icl_str + f"\nExample {idx+1}: \nInput: {trimmed_input}\nOutput: {output}\n\n"
                idx += 1

            messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                        {"role": "user", "content": user_prompt}]

    return messages

def gen_amazon_message(context, version='zero-shot', example_input=None, example_output=None):
    # version = 'zero-shot'
    categories_list = ", ".join([f"'{category}'" for category in amazon_mapping.values()])
    context = context[:1200] if len(context) > 1200 else context
    sys_prompt = f"You are an AI trained to categorize product reviews as either authentic or fraudulent. Your task is to analyze the review provided, consider its characteristics, and identify the most relevant category. Here are all of the categories: {categories_list}. You are to ONLY give a one word response, regarding the relevant category."
    user_prompt = f"Review description: {context.strip()}\nConsider its characteristics and give me the category of this review. Respond only with the category key (this is EITHER 'Authentic', 'Fraudulent'), without any additional text or explanation. Do not share your thinking process"
    if version == 'zero-shot':
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}]
    elif version == 'icl':
        icl_str = ""
        if example_input and example_output and len(example_input) == len(example_output):
            for idx, (inp, out) in enumerate(zip(example_input, example_output)):
                trimmed_input = inp[:800] if len(inp) > 800 else inp
                icl_str += f"\nExample {idx + 1}: \nInput: {trimmed_input}\nOutput: {out}\n\n"
        messages = [{"role": "system", "content": sys_prompt + f"\nHere are some examples to help you understand how to categorize products based on their descriptions:{icl_str}"},
                    {"role": "user", "content": user_prompt}]
    return messages

def gen_arxiv_message(context, version='zero-shot', example_input=None, example_output=None):
    version = 'zero-shot'
    categories_list = ", ".join([f"'{category}'" for category in amazon_mapping.values()])
    context = context[:1200] if len(context) > 1200 else context
    sys_prompt = f"You are an AI trained to categorize papers as either belonging to the category 'Authentic' or 'Fraudulent', that is there is one type of paper considered fraudulent, and you are identifying that type. Your task is to analyze the paper information provided, consider its characteristics, and identify the most relevant category."
    user_prompt = f"Review description: {context.strip()}\nConsider its characteristics and give me the category of this paper. Respond only with the category key (choose between, 'Authentic', 'Fraudulent'), without any additional text or explanation."
    if version == 'zero-shot':
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}]
    elif version == 'icl':
        icl_str = ""
        if example_input and example_output and len(example_input) == len(example_output):
            for idx, (inp, out) in enumerate(zip(example_input, example_output)):
                trimmed_input = inp[:800] if len(inp) > 800 else inp
                icl_str += f"\nExample {idx + 1}: \nInput: {trimmed_input}\nOutput: {out}\n\n"
        messages = [{"role": "system", "content": sys_prompt + f"\nHere are some examples to help you understand how to categorize products based on their descriptions:{icl_str}"},
                    {"role": "user", "content": user_prompt}]
    return messages

def gen_product_message(context, version='zero-shot', example_input=None, example_output=None):
    categories_list = ", ".join([f"'{category}'" for category in products_mapping.values()])
    context = context[:1200] if len(context) > 1200 else context
    sys_prompt = f"You are an AI trained to categorize products into specific categories based on their descriptions and characteristics. Your task is to analyze the product description provided, consider its characteristics, and identify the most relevant category among hundreds of possible categories. There are a total of 47 categories, including {categories_list}."
    user_prompt = f"Product description: {context.strip()}\nConsider its characteristics and give me the category of this product. Respond only with the category key (e.g., 'Electronics', 'Toys & Games'), without any additional text or explanation."
    if version == 'zero-shot':
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}]
    elif version == 'icl':
        icl_str = ""
        if example_input and example_output and len(example_input) == len(example_output):
            for idx, (inp, out) in enumerate(zip(example_input, example_output)):
                trimmed_input = inp[:800] if len(inp) > 800 else inp
                icl_str += f"\nExample {idx + 1}: \nInput: {trimmed_input}\nOutput: {out}\n\n"
        messages = [{"role": "system", "content": sys_prompt + f"\nHere are some examples to help you understand how to categorize products based on their descriptions:{icl_str}"},
                    {"role": "user", "content": user_prompt}]
    elif version == 'ex':
        icl_str = ""
        if example_input and example_output and len(example_input) == len(example_output):
            for idx, (inp, out) in enumerate(zip(example_input, example_output)):
                trimmed_input = inp[:800] if len(inp) > 800 else inp
                icl_str += f"\nProduct {idx + 1}: \n{trimmed_input}\n\n"
        messages = [{"role": "system", "content": sys_prompt + f"\nHere are some descriptions of other products that customers bought at the same time: {icl_str}"},
                    {"role": "user", "content": user_prompt}]
    else:
        raise ValueError(f"Unsupported version: {version}. Expected 'zero-shot', 'ex', or 'icl'.")
    return messages

def create_chat_message(context, version='zero-shot', example_input=None, example_output=None, dataset_name='ogbn-arxiv'):
    if dataset_name in ['ogbn-arxiv', 'arxiv_2023']:
        messages = gen_arxiv_message(context=context,
                                     version=version,
                                     example_input=example_input,
                                     example_output=example_output,
                                     )
    elif dataset_name == 'ogbn-products':
        messages = gen_product_message(context=context,
                                       version=version,
                                       example_input=example_input,
                                       example_output=example_output,
                                       )
    elif dataset_name == 'amazon':
        messages = gen_amazon_message(context=context,
                                       version=version,
                                       example_input=example_input,
                                       example_output=example_output,
                                       )
    return messages
if __name__ == "__main__":
    device = f'cuda'
    if args.llm_model == 'qwen':
        cach_path = 'Qwen/Qwen3-8B'
        cache_dir = os.path.join("/", "projects", "p32673", "AskGNN", "hf_cache")
        tokenizer = AutoTokenizer.from_pretrained(cach_path)
        llm_model = AutoModelForCausalLM.from_pretrained(
            cach_path,
            cache_dir=cache_dir,
            torch_dtype="auto",
            device_map="auto"
        )
    length = args.run
    base_model_type = args.llm_model
    api_key_list = []
    save_path_raw = 'your dict save path'
    file_path = f'results/Jun27-10_examples-800_train-MODEL.pkl'
    if args.dataset == 'ogbn-arxiv':
        categories = cs_categories
    elif args.dataset == 'ogbn-products':
        product_categories = {}
        for key, value in product_categories_raw.items():
            product_categories[key] = [products_mapping[value]]
        categories = product_categories
    elif args.dataset == 'arxiv_2023':
        arxiv24_categories = {}
        for key, value in arxiv24_categories_raw.items():
            arxiv24_categories[key] = [value]
        categories = arxiv24_categories
    elif args.dataset == 'amazon':
        categories = {0: "Fraudulent",
                      1: "Authentic"}
    # with open(file_path, 'rb') as file:
        # final_dict = pickle.load(file)

    # raw_data = final_dict['raw_data']
    # labels = final_dict['labels']
    # GNN_result = final_dict['GNN_result']
    method_name_map = {
        'map_dict': 'GNN+KNN ICL',
    }
    method_list = ['map_dict']
    final_result_dict = {}
    tmp_dict = {}
    # Oscar change: from 'map_dict_neighbors' to 'map_dict'
    # for key, _ in final_dict['map_dict'].items():
        # Oscar change: also changed this one - refer to original code to change back 
        # tmp_dict[key] = final_dict['map_dict'][key]
    map_dict = tmp_dict
    
    def get_method_acc(raw_data,  categories, map_dict,  true_labels):
        labels = true_labels
        result_dict = {}
        count = 0
        iter_dict = map_dict
        wrong_case_num = 0
        for key in iter_dict.keys():
            label = labels[key].item()
            raw_content = raw_data[key]
            if args.dataset == 'arxiv_2023':
                example_input = [raw_data[idx].replace('\n', '') for idx in iter_dict[key]]
            else:
                example_input = [raw_data[idx] for idx in iter_dict[key]]

            example_output = [categories[labels[idx].item()] for idx in map_dict[key]]
            messages = create_chat_message(context=raw_content, version='icl', example_input=example_input,
                                           example_output=example_output, dataset_name=args.dataset)

            with torch.no_grad():
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)

                generated_ids = llm_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                category_number = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                tail = category_number.rsplit("</think>", 1)[-1]
                category_number = "Fraudulent" if "fraudulent" in tail.lower() else "Authentic"            
                if category_number == "Fraudulent":
                    print(messages)
                result_dict[key] = {"label": label, "pred": category_number}
            if (categories[label] == category_number):
                count += 1
        result_dict['acc'] = count / (len(map_dict.keys()) - wrong_case_num)
        return result_dict

    with open(file_path, 'rb') as file:
        dict = pickle.load(file)
        categories = {0: "Fraudulent",
                      1: "Authentic"}
        final_result_dict = get_method_acc(dict['raw_data'], categories, dict['map_dict'], dict['labels'])
        save_dict_as_pickle(dictionary=final_result_dict, file_path='results/Jun27-10_examples-800_train')






