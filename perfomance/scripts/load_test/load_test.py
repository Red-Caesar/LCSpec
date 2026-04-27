import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Mapping

CURRENT_FILE_PATH = Path(__file__).parent.resolve()


def load_env_variables(args: argparse.Namespace) -> Mapping[str, str]:
    env = os.environ.copy()

    if args.run_id:
        env["RUN_ID"] = args.run_id

    env["HOST"] = args.url
    env["MODEL"] = args.model
    env["RUN_TIME"] = args.run_time
    env["OUTPUT_DISTRIBUTION"] = args.output_tokens_distribution
    env["ARTIFACTS_DIR"] = str(CURRENT_FILE_PATH.joinpath("artifacts"))
    env["LOG_FILE"] = str(CURRENT_FILE_PATH.joinpath(".logs"))
    env["DATASET_TYPE"] = args.dataset_type
    env["INPUT_TOKENS_DISTRIBUTION"] = args.input_tokens_distribution
    if args.use_tokenizer:
        env["TOKENIZER"] = args.model
    if args.spawn_rate:
        env["SPAWN_RATE"] = args.spawn_rate
    env["DURATION"] = args.run_time

    return env


def parse_users(users: str) -> List[str]:
    parameters = users.split(":")
    if len(parameters) == 1:
        return parameters
    elif len(parameters) == 2:
        min_users, max_users = parameters
        return list(map(str, range(int(min_users), int(max_users) + 1)))
    elif len(parameters) == 3:
        min_users, max_users, step = parameters
        return list(map(str, range(int(min_users), int(max_users) + 1, int(step))))
    else:
        raise ValueError(f"Incorrect format of users: {users}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Locust load test for LLM inference endpoints"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Target server host URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to test",
    )
    parser.add_argument(
        "--users",
        "-u",
        type=str,
        default="10",
        help="Number of concurrent users. Possible format min:max:step",
    )
    parser.add_argument(
        "--spawn-rate",
        "-r",
        type=str,
        default=None,
        help="Users spawned per second",
    )
    parser.add_argument(
        "--run-time",
        "-t",
        type=str,
        default="2s",
        help="Test duration (e.g., 60s, 5m, 1h)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="code",
        choices=["code", "sharegpt4", "summary"],
        help="The dataset type to use",
    )
    parser.add_argument(
        "--input-tokens-distribution",
        type=str,
        default="const(100)",
        help="""
        The params for prompt tokens:
            const(n)
            normal(mean, std)
        """,
    )
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="The tokenizer will be used for prompts truncation",
    )
    parser.add_argument(
        "--output-tokens-distribution",
        type=str,
        default="const(100)",
        help="""
        The params for output tokens:
            const(n)
            normal(median, std)
        """,
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier",
    )
    parser.add_argument(
        "--web-ui",
        action="store_true",
        help="Run with web UI (overrides --headless)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8110,
        help="Port for web UI",
    )

    args = parser.parse_args()
    env = load_env_variables(args)

    locust_file = Path(__file__).parent / "locust.py"
    logs_dir = Path(__file__).parent / ".logs"
    logs_dir.mkdir(exist_ok=True)

    users_array = parse_users(args.users)
    if len(users_array) > 1 and args.web_ui:
        raise ValueError("Web UI is not supported for multiple users")

    for users in users_array:
        env["USERS"] = users
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"locust_{timestamp}.log"
        cmd = [
            "locust",
            "-f",
            str(locust_file),
            "--users",
            users,
            "--run-time",
            args.run_time,
            "--host",
            args.url,
            "--logfile",
            str(log_file),
            # "--skip-log-setup",
        ]

        if args.web_ui:
            cmd.extend(["--web-port", str(args.web_port)])
            print(f"\nStarting Locust with web UI at http://localhost:{args.web_port}")
        else:
            cmd.append("--headless")
            cmd.append("--only-summary")

        if args.spawn_rate:
            cmd.append(f"--spawn-rate {args.spawn_rate}")

        Path(CURRENT_FILE_PATH.joinpath("artifacts")).mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(cmd, env=env)
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n\nLoad test interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\nError running load test: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
