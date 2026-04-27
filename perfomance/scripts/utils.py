import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import libtmux
import requests
import yaml
from tqdm import tqdm

LOG_PATH = Path(".logs")
PATH_TO_VENV = Path(__file__).parent.parent.parent.resolve() / ".venv" / "bin" / "activate"
DATASET_CACHE_DIR = Path(__file__).parent / "tokenized_datasets"


def setup_logger(
    output_dir: Union[str, Path] = LOG_PATH, log_name: str = "test"
) -> logging.Logger:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / f"{log_name}.log"
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_prompts(dataset_type: str, cache_dir: Path) -> List[str]:
    from datasets import load_dataset

    logger = logging.getLogger("utils")
    cache_file = cache_dir / f"{dataset_type}.json"
    if cache_file.exists():
        logger.info(f"Loading cached prompts from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    if dataset_type == "code":
        dataset = load_dataset("google-research-datasets/mbpp", "full")
        prompts = list(dataset["test"]["text"])
    elif dataset_type == "summary":
        dataset = load_dataset("ChicagoHAI/CaseSumm", split="train", trust_remote_code=True)
        system_prompt = "## TASK: Make a summary of the following text:\n\n ## TEXT: "
        prompts = [system_prompt + doc for doc in dataset["opinion"]]
    elif dataset_type == "chat":
        dataset = load_dataset("shibing624/sharegpt_gpt4", split="train")
        prompts = [conv[0]["value"] for conv in dataset["conversations"]]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(prompts, f)
    logger.info(f"Saved prompts cache to {cache_file}")
    return prompts


def load_prompt_token_ids(
    dataset_type: str, tokenizer: Any, cache_dir: Path
) -> List[List[int]]:
    logger = logging.getLogger("utils")
    tokens_cache_file = cache_dir / f"{dataset_type}_tokens.json"
    if tokens_cache_file.exists():
        logger.info(f"Loading cached token ids from {tokens_cache_file}")
        with open(tokens_cache_file) as f:
            return json.load(f)

    prompts = load_raw_prompts(dataset_type, cache_dir)
    logger.info("Tokenizing prompts...")
    tokens_list = [tokenizer.encode(prompt) for prompt in prompts]
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(tokens_cache_file, "w") as f:
        json.dump(tokens_list, f)
    logger.info(f"Saved token ids cache to {tokens_cache_file}")
    return tokens_list


def prepare_input_tokens(input_tokens_str: str | None) -> List[int | None]:
    """
    Accepts: a single int, ``min:max``, or ``min:max:step``.
    """
    if input_tokens_str is None:
        return [None]
    parts = input_tokens_str.split(":")
    if len(parts) > 3:
        raise ValueError("Expected format: <value>, <min:max>, or <min:max:step>")
    if len(parts) == 1:
        return [int(parts[0])]
    elif len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    else:
        return list(range(int(parts[0]), int(parts[1]) + 1, int(parts[2])))


def normalize_model_name(model_path: str) -> str:
    path = model_path.rstrip("/")
    return path.split("/")[-1]


def build_vllm_commands(setup: Dict, extra_env: Optional[Dict] = None) -> List[str]:
    server_args = setup.get("server_args", {})
    env_args = dict(setup.get("env", {}))
    if extra_env:
        env_args.update(extra_env)
    env_str = " ".join(f"{k}={v}" for k, v in env_args.items())

    args_str = " ".join(
        f"--{k} {v}"
        for k, v in server_args.items()
        if k not in ["model", "speculative_config", "attention_config"]
    )
    model_name = server_args["model"]
    if "speculative_config" in server_args:
        args_str += f" --speculative_config '{json.dumps(server_args['speculative_config'])}'"
    if "attention_config" in server_args:
        args_str += f" --attention_config '{json.dumps(server_args['attention_config'])}'"

    return [
        f"source {PATH_TO_VENV}",
        f"{env_str} vllm serve {model_name} {args_str}",
    ]


def wait_for_server(
    url: str,
    delay: int = 5,
    max_retries: int = 60,
    warmup_delay: int = 0,
) -> None:
    logger = logging.getLogger("utils")
    for _ in tqdm(range(max_retries), desc=f"Waiting for server at {url}"):
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("Server is ready.")
                if warmup_delay:
                    time.sleep(warmup_delay)
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    raise TimeoutError(f"Server did not become ready after {max_retries * delay}s")


def start_vllm_in_tmux(
    session_name: str, setup: Dict, extra_env: Optional[Dict] = None
) -> None:
    logger = logging.getLogger("utils")
    server = libtmux.Server()
    if server.has_session(session_name):
        logger.info(f"Killing existing tmux session '{session_name}'")
        server.kill_session(session_name)

    commands = build_vllm_commands(setup, extra_env=extra_env)
    logger.info(f"Starting vLLM in tmux session '{session_name}'")
    session = server.new_session(session_name)
    pane = session.active_window.active_pane
    for cmd in commands:
        pane.send_keys(cmd)


def kill_tmux_session(session_name: str) -> None:
    server = libtmux.Server()
    if server.has_session(session_name):
        server.kill_session(session_name)