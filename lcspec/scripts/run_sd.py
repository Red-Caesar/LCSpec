import argparse
import contextlib
import gc
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Any

import ray
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from lcspec.scripts.utils import load_config, setup_logger
from vllm.v1.metrics.reader import Metric
from transformers import AutoTokenizer


logger = setup_logger(log_name="sd_experiments")


@dataclass
class SDMetrics:
    main_model: str
    speculative_model: str | None
    dataset_type: str
    num_prompts: int
    time_taken: float
    timestamp: str
    mean_acceptance_length: float | None = None
    acceptance_rates: List[float] | None = None
    num_drafts: int | None = None
    input_tokens: int | None = None

    def to_dict(self):
        return asdict(self)

    def save_to_json(self, output_path: Path):
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_spec_acceptance_metrics(
    metrics: list[Metric], k: int
    ) -> Dict[str, Any]:
    num_drafts = 0
    num_accepted = 0
    acceptance_counts = [0] * k
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            num_accepted += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]
    acceptance_rate_per_pos = [
        count / num_drafts for count in acceptance_counts
    ]
    mean_acceptance_length = 1 + (num_accepted / num_drafts)
    return {
        "num_drafts": num_drafts,
        "num_accepted": num_accepted,
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
        "mean_acceptance_length": mean_acceptance_length,
    }


def prepare_prompts(
    dataset_type: str,
    num_prompts: int,
    input_tokens: int | None = None,
    tokenizer: AutoTokenizer | None = None,
    ) -> List[int]:

    if dataset_type == "code":
        dataset = load_dataset("google-research-datasets/mbpp", "full")
        prompts = dataset["test"]["text"]
    elif dataset_type == "summary":
        dataset = load_dataset("ChicagoHAI/CaseSumm", split="train", trust_remote_code=True)
        system_prompt = "## TASK: Make a summary of the following text:\n\n ## TEXT: "
        prompts = [system_prompt + doc for doc in dataset["opinion"]]
    elif dataset_type == "chat":
        dataset = load_dataset("shibing624/sharegpt_gpt4", split="train")
        prompts = [conv[0]["value"] for conv in dataset["conversations"]]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if input_tokens:
        # Make an approximation that 1 token is about 3 characters
        # prompts = [tokens[:input_tokens*3] for tokens in prompts]
        tokens_list = [tokenizer.encode(prompt) for prompt in prompts]
        std = 100
        min_prompt_tokens = max(1, input_tokens - std)
        limited_tokens = [tokens[:input_tokens] for tokens in tokens_list if len(tokens) > min_prompt_tokens]
        prompts = tokenizer.batch_decode(limited_tokens)


    if num_prompts == -1:
        return prompts
    return prompts[:num_prompts]


def cleanup_vllm(llm: LLM):
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    logger.info("Successfully delete the llm pipeline and free the GPU memory.")



def run_offline_vllm(
    server_args: Dict,
    dataset_type: str,
    num_prompts: int,
    output_dir: Path,
    input_tokens: int | None,
    output_tokens: int,
) -> SDMetrics:
    logger.info(f"Initializing vLLM with config: {server_args}")

    llm = LLM(**server_args)
    sampling_params = SamplingParams(temperature=0, max_tokens=output_tokens)
    tokenizer = AutoTokenizer.from_pretrained(server_args.get("model"))
    prompts = prepare_prompts(dataset_type, num_prompts, input_tokens, tokenizer)
    # messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    batch_size = 32
    batch_array = []
    messages = []
    for i, prompt in enumerate(prompts):
        messages.append([{"role": "user", "content": prompt}])
        if (i + 1) % batch_size == 0:
            batch_array.append(messages)
            messages = []

    if messages is not None:
        batch_array.append(messages)

    start = time.time()
    for batch in batch_array:
        output = llm.chat(batch, sampling_params, use_tqdm=False)
    # for message in tqdm(messages, desc="Generating outputs"):
    #     output = llm.chat(message, sampling_params)
    #     outputs.append(output[0])
    # for prompt in tqdm(prompts, desc="Generating outputs"):
    #     output = llm.generate(prompt, sampling_params, use_tqdm=False)
    #     outputs.append(output[0])
    end = time.time()

    time_taken = end - start

    spec_config = server_args.get("speculative_config")
    mean_acceptance_length = None
    acceptance_rates = None
    num_drafts = None

    if spec_config:
        num_spec_tokens = spec_config["num_speculative_tokens"]
        metrics = get_spec_acceptance_metrics(llm.get_metrics(), k=num_spec_tokens)
        mean_acceptance_length = metrics["mean_acceptance_length"]
        acceptance_rates = metrics["acceptance_rate_per_pos"]
        num_drafts = metrics["num_drafts"]

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    metrics = SDMetrics(
        main_model=server_args["model"],
        speculative_model=spec_config.get("model", "MTP") if spec_config else None,
        dataset_type=dataset_type,
        num_prompts=len(prompts),
        time_taken=time_taken,
        mean_acceptance_length=mean_acceptance_length,
        acceptance_rates=acceptance_rates,
        num_drafts=num_drafts,
        timestamp=timestamp.replace("_", " "),
        input_tokens=input_tokens,
    )

    output_file = output_dir / f"sd_results_{timestamp}.json"
    metrics.save_to_json(output_file)
    logger.info(f"Results saved to {output_file}")
    cleanup_vllm(llm)
    return metrics

def prepare_input_tokens(input_tokens_str: str | None) -> List[int]:
    if input_tokens_str is None:
        return [None]
    parts = input_tokens_str.split(":")
    if len(parts) > 3:
        raise ValueError("Invalid input tokens format. Expected format: <min:max:step>")

    if len(parts) == 1:
        return [int(parts[0])]
    elif len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    else:
        return list(range(int(parts[0]), int(parts[1]) + 1, int(parts[2])))


def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["code", "summary", "chat"],
        required=True,
        help="Dataset type to use",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=-1,
        help="Number of prompts to use (-1 for all)",
    )
    parser.add_argument(
        "--setup_type",
        type=str,
        choices=["single_setup", "few_setups"],
        required=True,
        help="Which setup to use from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sd_experiments",
        help="Directory to save results",
    )
    parser.add_argument(
        "--input_tokens",
        type=str,
        default=None,
        help="Number of tokens to use for input. It could be a range: <min:max:step>"
    )
    parser.add_argument(
        "--output_tokens",
        type=int,
        default=256,
        help="Number of tokens for output"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_tokens_list = prepare_input_tokens(args.input_tokens)
    logger.info(f"Using input tokens: {input_tokens_list}")
    for input_tokens in input_tokens_list:
        logger.info(f"Running experiments with input tokens: {input_tokens}")
        if args.setup_type == "single_setup":
            run_offline_vllm(
                server_args=config["single_setup"]["server_args"],
                dataset_type=args.dataset,
                num_prompts=args.num_prompts,
                output_dir=output_dir,
                input_tokens=input_tokens,
                output_tokens=args.output_tokens,
            )
            main_model = config["single_setup"]["server_args"]["model"]
            speculative_model = (
                config["single_setup"]["server_args"]
                .get("speculative_config", {})
                .get("model", "None")
            )
            logger.info(
                f"Single setup completed: {main_model} {'with ' + speculative_model}"
            )
        else:
            for setup in tqdm(config["few_setups"]):
                main_model = setup["server_args"]["model"]
                speculative_model = (
                    setup["server_args"].get("speculative_config", {}).get("model", "None")
                )
                try:
                    run_offline_vllm(
                        server_args=setup["server_args"],
                        dataset_type=args.dataset,
                        num_prompts=args.num_prompts,
                        output_dir=output_dir,
                        input_tokens=input_tokens,
                        output_tokens=args.output_tokens,
                    )
                    logger.info(
                        f"Setup completed: {main_model} {'with ' + speculative_model}"
                    )
                except Exception as e:
                    error_msg = (
                        f"Setup failed: {main_model} {'with ' + speculative_model}:\n"
                        f"{str(e)}\n"
                        f"Traceback:\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    continue


if __name__ == "__main__":
    main()
