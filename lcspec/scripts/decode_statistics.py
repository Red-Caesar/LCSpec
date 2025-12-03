import argparse
import requests
import json
import os
import pickle
import random
from typing import List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import time

# python decode_statistics.py --max-tokens 10 --model-name "meta-llama/Llama-3.1-8B-Instruct" --num-prompts 1 --context-len 4000
@dataclass
class Statistics:
    num_prompts: int
    context_len: int
    total_prompts: int
    max_tokens: int
    decodes_array: List[int]
    median_num_decodes: int

def choose_tokenized_prompt(
    context_len: int,
    num_prompts: int,
    model_name: str
    ) -> Tuple[List[str], int]:
    dataset = load_dataset(
        "ChicagoHAI/CaseSumm", split="train", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    system_prompt = "## TASK: Make a summary of the following text:\n\n ## TEXT: "
    prompts = [system_prompt + doc for doc in dataset["opinion"]]
    
    if os.path.exists("case_summ_tokenized.pickle"):
        print("Load tokenized prompts...")
        with open("case_summ_tokenized.pickle", "rb") as f:
            tokenized_prompts = pickle.load(f)
    else:
        print("Starting prompt tokenization...")
        tokenized_prompts = [tokenizer.tokenize(text) for text in prompts]
        print("Ended prompt tokenization...")
        with open("case_summ_tokenized.pickle", "wb") as f:
            pickle.dump(tokenized_prompts, f)
    
    allowed_prompts = []
    for i, tokens in enumerate(tokenized_prompts):
        if len(tokens) >= context_len:
            prompt = tokenizer.convert_tokens_to_string(tokens[:context_len])
            allowed_prompts.append(prompt)

    # return allowed_prompts[:num_prompts]
    return random.choices(allowed_prompts, k=num_prompts), len(allowed_prompts)

def send_prompt_to_api(prompt: str, max_tokens: int, url: str, model_name: str) -> int:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    
    try:
        chunk_count = 0
        response = requests.post(f"{url}/v1/chat/completions", headers=headers, data=json.dumps(payload))
        for chunk in response.iter_lines():
            if chunk:
                decoded_chunk = chunk.decode("utf-8")
                # print(f"Response chunk: {decoded_chunk}")
                chunk_count += 1
        return chunk_count - 3 # exclude "" chunk, prefill and "data: [DONE]"
    except Exception as e:
        print(e)
        return 0

def save_histogram(
    decodes_array: List[int],
    max_tokens: int,
    context_len: int,
    output_dir: str,
    output_file: str): 
    plt.figure(figsize=(10, 6))
    plt.hist(decodes_array, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
    
    median_val = np.median(decodes_array)
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    
    plt.title(f'Histogram of decodes on {max_tokens} max_tokens and {context_len} context_len')
    plt.legend()
    plt.grid(True, alpha=0.3)

    hist_file = os.path.join(output_dir, f"{output_file}_histogram.png")
    plt.tight_layout()
    plt.savefig(hist_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Collecting decode statistics")
    parser.add_argument("--max-tokens", type=int, default=10, help="Maximum tokens for completion")
    parser.add_argument("--url", type=str, default="http://0.0.0.0:8000", help="Base URL of the API endpoint")
    parser.add_argument("--model-name", type=str, default="dummy_model", help="Model name to use")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts to test")
    parser.add_argument("--context-len", type=int, default=100, help="Number of tokens per prompt")
    parser.add_argument("--output-file", type=str, default="test")
    args = parser.parse_args()  
    
    prompts, total_prompts = choose_tokenized_prompt(args.context_len, args.num_prompts, args.model_name)
    chunk_counts = []
    for prompt in tqdm(prompts):
        chunk_count = send_prompt_to_api(prompt, args.max_tokens, args.url, args.model_name)
        chunk_counts.append(chunk_count)
        # time.sleep(2)

    stats = Statistics(
        num_prompts=len(prompts),
        context_len=args.context_len,
        total_prompts=total_prompts,
        max_tokens=args.max_tokens,
        decodes_array=chunk_counts,
        median_num_decodes=int(np.median(chunk_counts)) if chunk_counts else 0
    )

    for field, value in stats.__dict__.items():
        print(f"{field}: {value}")

    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", f"{args.output_file}.json")
    with open(output_file, "w") as f:
       json.dump(asdict(stats), f)
    
    save_histogram(
        decodes_array=chunk_counts,
        max_tokens=args.max_tokens,
        context_len=args.context_len,
        output_dir="results",
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
