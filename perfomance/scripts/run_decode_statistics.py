import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from perfomance.scripts.utils import (
    DATASET_CACHE_DIR,
    kill_tmux_session,
    load_config,
    load_prompt_token_ids,
    normalize_model_name,
    prepare_input_tokens,
    setup_logger,
    start_vllm_in_tmux,
    wait_for_server,
)

logger = setup_logger(log_name="decode_statistics")


@dataclass
class Statistics:
    run_id: str
    num_prompts: int
    input_tokens: int
    total_prompts: int
    max_tokens: int
    decodes_array: List[int]
    median_num_decodes: int


def choose_tokenized_prompt(
    input_tokens: int,
    num_prompts: int,
    model_name: str,
) -> Tuple[List[str], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cache_dir = DATASET_CACHE_DIR / normalize_model_name(model_name)
    tokens_list = load_prompt_token_ids("summary", tokenizer, cache_dir)
    allowed_prompts = [
        tokenizer.decode(tokens[:input_tokens])
        for tokens in tokens_list
        if len(tokens) >= input_tokens
    ]
    return random.choices(allowed_prompts, k=num_prompts), len(allowed_prompts)


def send_prompt_to_api(prompt: str, max_tokens: int, url: str, model_name: str) -> int:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        chunk_count = 0
        response = requests.post(
            f"{url}/v1/chat/completions", headers=headers, data=json.dumps(payload)
        )
        for chunk in response.iter_lines():
            if chunk:
                chunk_count += 1
        return chunk_count - 3  # exclude empty chunk, prefill, and "data: [DONE]"
    except Exception as e:
        logger.error(e)
        return 0


def save_histogram(
    decodes_array: List[int],
    max_tokens: int,
    context_len: int,
    run_id: str,
    output_dir: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(decodes_array, bins="auto", alpha=0.7, color="skyblue", edgecolor="black")

    median_val = np.median(decodes_array)
    plt.axvline(
        median_val, color="green", linestyle="--", linewidth=2,
        label=f"Median: {median_val:.2f}",
    )

    plt.title(
        f"[{run_id}] Decodes distribution | max_tokens={max_tokens}, context_len={context_len}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    hist_file = os.path.join(output_dir, f"{run_id}_histogram.png")
    plt.savefig(hist_file, dpi=300, bbox_inches="tight")
    plt.close()


def run_setup(setup: dict, args: argparse.Namespace, input_tokens_list: List[int]) -> None:
    run_id = setup["run_id"]
    server_args = setup["server_args"]
    model_name = server_args["model"]
    port = server_args.get("port", "8000")
    url = f"http://localhost:{port}"

    start_vllm_in_tmux(run_id, setup, extra_env={"VLLM_DISABLE_COMPILE_CACHE": "1"})
    wait_for_server(url)

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        for input_tokens in input_tokens_list:
            prompts, total_prompts = choose_tokenized_prompt(
                input_tokens, args.num_prompts, model_name
            )
            chunk_counts = []
            for prompt in tqdm(prompts, desc=f"[{run_id}@{input_tokens}tok] collecting decodes"):
                chunk_counts.append(
                    send_prompt_to_api(prompt, args.max_tokens, url, model_name)
                )

            stats = Statistics(
                run_id=run_id,
                num_prompts=len(prompts),
                input_tokens=input_tokens,
                total_prompts=total_prompts,
                max_tokens=args.max_tokens,
                decodes_array=chunk_counts,
                median_num_decodes=int(np.median(chunk_counts)) if chunk_counts else 0,
            )

            output_file = os.path.join(args.output_dir, f"{run_id}_{input_tokens}.json")
            with open(output_file, "w") as f:
                json.dump(asdict(stats), f, indent=2)
            logger.info(f"Results saved to {output_file}")

            save_histogram(
                decodes_array=chunk_counts,
                max_tokens=args.max_tokens,
                context_len=input_tokens,
                run_id=f"{run_id}_{input_tokens}",
                output_dir=args.output_dir,
            )
    finally:
        logger.info(f"Killing tmux session '{run_id}'")
        kill_tmux_session(run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect decode statistics via vLLM")
    parser.add_argument("--config", type=str, required=True, help="Path to setups YAML config")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument(
        "--input-tokens",
        type=str,
        required=True,
        help="Number of tokens to use for input. It could be a range: <min:max:step>",
    )
    parser.add_argument("--output-dir", type=str, default="results/decode_statistics")
    args = parser.parse_args()

    input_tokens_list = prepare_input_tokens(args.input_tokens)
    logger.info(f"Running over input token values: {input_tokens_list}")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)

    config = load_config(args.config)
    for setup in config["setups"]:
        logger.info(f"Running setup: {setup['run_id']}")
        run_setup(setup, args, input_tokens_list)


if __name__ == "__main__":
    main()
