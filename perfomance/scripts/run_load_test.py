import argparse
import subprocess
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from perfomance.scripts.utils import (
    kill_tmux_session,
    load_config,
    normalize_model_name,
    setup_logger,
    start_vllm_in_tmux,
    wait_for_server,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
CURRENT_DIRECTORY_PATH = Path(__file__).parent.resolve()
logger = setup_logger(log_name="run_load_test_summary")


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
        input_str = f'"normal({context_len},{std})"'
        inputs_list.append(input_str)
    return inputs_list


def run_evaluation(
    setup: Dict[str, str],
    input_tokens: Optional[str],
) -> None:
    dir_log = Path(__file__).parent.parent / ".logs"
    dir_log.mkdir(parents=True, exist_ok=True)

    model_name = normalize_model_name(setup["vllm"]["server_args"]["model"])
    model_name = model_name.replace("/", "_").replace(".", "_")

    spec_config = setup["vllm"]["server_args"].get("speculative_config", {})
    current_setup = model_name
    suffix_run_id = ""
    if spec_config.get("model"):
        speculator_name = normalize_model_name(spec_config["model"]).replace("/", "_").replace(".", "_")
        speculator_method = spec_config.get("method", "")
        draft_name = speculator_name + "#" + speculator_method
        num_speculative_tokens = spec_config.get("num_speculative_tokens", "0")
        current_setup += f"_{draft_name}_{num_speculative_tokens}"
        suffix_run_id = "#" + draft_name + "#" + str(num_speculative_tokens)

    port = setup["vllm"]["server_args"].get("port", "8000")
    base_url = f"http://localhost:{port}"

    logger.info("Starting vllm server")
    start_vllm_in_tmux(current_setup, setup["vllm"])
    try:
        wait_for_server(base_url, delay=10, max_retries=180, warmup_delay=10)
    except TimeoutError:
        kill_tmux_session(current_setup)
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
    kill_tmux_session(current_setup)


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
