import argparse
import json
import subprocess
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import libtmux
import requests
from tqdm import tqdm

from lcspec.scripts.utils import load_config, setup_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
CURRENT_DIRECTORY_PATH = Path(__file__).parent.resolve()
ROOT_DIRECTORY_PATH = Path(__file__).parent.parent.parent.resolve()
PATH_TO_VENV = ROOT_DIRECTORY_PATH.joinpath(".venv", "bin", "activate")
logger = setup_logger(log_name="run_load_test_summary")


def wait_for_server(url: str, max_retries: int = 180, delay: int = 10) -> bool:
    for i in tqdm(range(max_retries), desc=f"Waiting for server at {url} to start..."):
        try:
            response = requests.get(url + "/health")
            if response.status_code == 200:
                time.sleep(10)  # Warm up time for the server
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    return False


def run_background_process(
    command: str, output_dir: Path, log_name: str
) -> subprocess.Popen:
    with open(output_dir / log_name, "w") as process_out:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=process_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process


def create_vllm_command(setup: Dict[str, str]) -> List[str]:
    server_args = setup.get("server_args", {})
    env_args = setup.get("env", {})
    env_args = " ".join([f"{k}={v}" for k, v in env_args.items()])
    args_str = " ".join(
        [
            f"--{k} {v}"
            for k, v in server_args.items()
            if k not in ["model", "speculative_config"]
        ]
    )
    model_name = server_args["model"]
    if "speculative_config" in server_args:
        spec_config = json.dumps(server_args["speculative_config"])
        args_str += f" --speculative_config '{spec_config}'"
    commands = [
        f"source {PATH_TO_VENV}",
        f"{env_args} vllm serve {model_name} {args_str}",
    ]
    return commands


def create_load_test_command(
    load_test_args: Dict[str, str],
    model_name: str,
    suffix_run_id: str,
    vllm_url: str,
) -> str:
    args_str = " ".join(
        [f"--{k} {v}" for k, v in load_test_args.items() if k not in ["run-id"]]
    )
    full_run_id = f"{load_test_args['run-id']}{suffix_run_id}"
    cmd = f"cd {CURRENT_DIRECTORY_PATH / 'load_test'} && python3 load_test.py "
    return (
        cmd
        + f"{args_str} --model {model_name} --use-tokenizer --url {vllm_url} --run-id {full_run_id}"
    )


def create_input_distribution(input_tokens: str) -> List[str]:
    input_values = input_tokens.split(":")
    assert len(input_values) == 3, "Input tokens should be in format min:max:step"
    min_tokens, max_tokens, step = map(int, input_values)

    inputs_list = []
    std = 100
    for context_len in range(min_tokens, max_tokens, step):
        min_prompt_len = max(0, context_len - std)
        input_str = f'"normal({context_len},{min_prompt_len})"'
        inputs_list.append(input_str)
    return inputs_list


def run_evaluation(
    setup: Dict[str, str],
    input_tokens: Optional[str],
) -> None:
    """
    Run evaluation for a given setup. Steps:
    1. Start vllm server with specified model and parameters.
    2. Wait for the server to be ready.
    3. Run load test with specified load values (rps, vus).
    4. Log results and clean up.
    """
    dir_log = Path(__file__).parent.parent / ".logs"
    vllm_commands = create_vllm_command(setup["vllm"])

    model_name = setup["vllm"]["server_args"]["model"]

    if model_name[-1] == "/":
        model_name = model_name[:-1]
    model_name = model_name.split("/")[-1].replace("/", "_").replace(".", "_")

    speculator_name = (
        setup["vllm"]["server_args"].get("speculative_config", {}).get("model", "")
    )

    current_setup = f"{model_name}"
    suffix_run_id = ""
    if speculator_name:
        if speculator_name[-1] == "/":
            speculator_name = speculator_name[:-1]

        speculator_name = (
            speculator_name.split("/")[-1].replace("/", "_").replace(".", "_")
        )
        speculator_method = (
            setup["vllm"]["server_args"].get("speculative_config", {}).get("method", "")
        )
        draft_name = speculator_name + "#" + speculator_method
        num_spec_tokens = (
            setup["vllm"]["server_args"]
            .get("speculative_config", {})
            .get("num_speculative_tokens", "")
        )

        current_setup += f"_{draft_name}_{num_spec_tokens}"
        num_speculative_tokens = (
            setup["vllm"]["server_args"]
            .get("speculative_config", {})
            .get("num_speculative_tokens", "0")
        )
        suffix_run_id = "#" + draft_name + "#" + str(num_speculative_tokens)

    logger.info("Starting vllm server")
    server = libtmux.Server()
    if server.has_session(current_setup):
        server.kill_session(current_setup)

    session = server.new_session(current_setup)
    window = session.attached_window
    pane = window.attached_pane
    for cmd in vllm_commands:
        pane.send_keys(cmd)

    port = setup["vllm"]["server_args"].get("port", "8000")
    base_url = f"http://localhost:{port}"
    if not wait_for_server(base_url):
        server.kill_session(current_setup)
        raise RuntimeError(f"Server at {base_url} did not start in time")
    logger.info("Started vllm server.")

    if input_tokens:
        inputs_list = create_input_distribution(input_tokens)
    else:
        inputs_list = [setup["load_test"].get("input-tokens-distribution", None)]
    for input_distribution in inputs_list:
        if input_distribution is not None:
            setup["load_test"]["input-tokens-distribution"] = input_distribution

        load_test_command = create_load_test_command(
            setup["load_test"],
            setup["vllm"]["server_args"]["model"],
            suffix_run_id,
            base_url,
        )
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        log_name = f"run_load_test_{current_setup}_{timestamp}.log"
        logger.info(
            f"Running load test with setup {current_setup} and input distribution {input_distribution}"
        )
        load_test_process = run_background_process(load_test_command, dir_log, log_name)
        load_test_process.wait()
        logger.info(
            f"Load test completed for setup {current_setup} and input distribution {input_distribution}"
        )
        logger.info("-" * 80)
    server.kill_session(current_setup)


def main():
    parser = argparse.ArgumentParser(description="Run server evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input-tokens", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    for setup in tqdm(config["setups"]):
        try:
            run_evaluation(setup, args.input_tokens)
            logger.info("Setup completed successfully")
        except Exception as e:
            error_msg = f"Setup failed:\n{str(e)}\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            continue


if __name__ == "__main__":
    main()
