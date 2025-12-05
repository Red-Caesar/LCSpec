import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from locust import HttpUser, TaskSet, events, task
from locust.env import Environment

from lcspec.scripts.load_test.custom_datasets import DatasetFactory


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    global last_progress_time
    last_progress_time = 0

    dataset_type = os.environ.get("DATASET_TYPE", "code")
    input_tokens_distribution = os.environ.get(
        "INPUT_TOKENS_DISTRIBUTION", "const(100)"
    )
    tokenizer_name = os.environ.get("TOKENIZER", None)
    dataset = DatasetFactory(dataset_type, input_tokens_distribution, tokenizer_name)
    global prompts
    prompts = dataset.prompts


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    global test_start_time, test_duration_seconds
    test_start_time = time.time()

    run_time_str = os.environ.get("DURATION", "404s")
    if run_time_str.endswith("s"):
        test_duration_seconds = int(run_time_str[:-1])
    elif run_time_str.endswith("m"):
        test_duration_seconds = int(run_time_str[:-1]) * 60
    elif run_time_str.endswith("h"):
        test_duration_seconds = int(run_time_str[:-1]) * 3600
    else:
        test_duration_seconds = int(run_time_str)

    print(f"\n{'=' * 60}")
    print(
        f"Load test started at {datetime.fromtimestamp(test_start_time).strftime('%H:%M:%S')}"
    )
    print(f"Duration: {run_time_str}")
    print(f"{'=' * 60}\n")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    global last_progress_time

    if test_start_time and test_duration_seconds:
        current_time = time.time()

        if current_time - last_progress_time >= 1:
            last_progress_time = current_time
            elapsed = current_time - test_start_time
            remaining = max(0, test_duration_seconds - elapsed)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
            progress = (
                (elapsed / test_duration_seconds) * 100
                if test_duration_seconds > 0
                else 0
            )
            print(
                f"\rProgress: {progress:.1f}% | Elapsed: {elapsed_str} | Remaining: {remaining_str} ",
                end="",
                flush=True,
            )


class LLMInferenceTasks(TaskSet):
    def on_start(self):
        """Called when a simulated user starts."""
        prompt_index = np.random.choice(np.array(range(len(prompts))))
        self.prompt = prompts[prompt_index]

        output_distribution = os.environ.get("OUTPUT_DISTRIBUTION", "const(100)")
        distribution, params = output_distribution[:-1].split("(")
        if distribution == "const":
            self.max_tokens = int(params)
        elif distribution == "normal":
            mean, std = list(map(int, params.split(",")))
            self.max_tokens = np.random.normal(mean, std)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    @task
    def generate_text(self):
        """Main task: send inference request and collect metrics."""
        request_payload = {
            "model": os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=request_payload,
            catch_response=True,
            name="chat_completion",
        ) as response:
            if response.status_code == 200:
                try:
                    response.success()
                except json.JSONDecodeError as e:
                    response.failure(f"Invalid JSON response: {str(e)}")
            else:
                response.failure(f"HTTP {response.status_code}")


class LLMUser(HttpUser):
    tasks = [LLMInferenceTasks]
    # wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = os.environ.get("HOST", "http://localhost:8000")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """
    Called when the load test stops.
    Saves metrics to artifacts folder.
    """
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "./artifacts"))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = os.environ.get("RUN_ID", f"run_{timestamp}")
    run_dir = artifacts_dir / f"{run_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_dist, input_params = os.environ.get("INPUT_TOKENS_DISTRIBUTION", "")[
        :-1
    ].split("(")
    output_dist, output_params = os.environ.get("OUTPUT_DISTRIBUTION", "")[:-1].split(
        "("
    )
    config = {
        "run_id": run_id,
        "server": {
            "host": os.environ.get("HOST", "http://localhost:8000"),
            "model": os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        },
        "workload": {
            "load": f"users={os.environ.get('USERS', 0)}",
            "dataset": f"{os.environ.get('DATASET_TYPE', None)}",
            "distribution": input_dist,
            "context_len": input_params.split(",")[0],
            "duration": os.environ.get("DURATION", "404s"),
            "output_tokens": output_params.split(",")[0],
        },
        "test_metadata": {
            "start_time": timestamp,
            "spawn_rate": os.environ.get("SPAWN_RATE", None),
        },
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if environment.runner and hasattr(environment.runner, "stats"):
        stats = environment.runner.stats

        if stats.entries:
            (_, stat) = list(stats.entries.items())[0]

            locust_stats = {
                "summary": {
                    "num_requests": stat.num_requests,
                    "num_failures": stat.num_failures,
                    "end-to-end": {
                        "avg": stat.avg_response_time,
                        "min": stat.min_response_time,
                        "max": stat.max_response_time,
                        "median": stat.median_response_time,
                        "p90": stat.get_response_time_percentile(0.90),
                        "p95": stat.get_response_time_percentile(0.95),
                        "p99": stat.get_response_time_percentile(0.99),
                    },
                    "total_rps": stat.total_rps,
                }
            }
        else:
            locust_stats = {"summary": {}}

        with open(run_dir / "summary.json", "w") as f:
            json.dump(locust_stats, f, indent=2)

    print(f"\n\n{'=' * 60}")
    print("Load test completed!")
    print(f"Results saved to: {run_dir}")
    print(f"{'=' * 60}")
    print(f"Total requests: {locust_stats['summary'].get('num_requests')}")
    print(f"Failed: {locust_stats['summary'].get('num_failures')}")
    print(f"{'=' * 60}")
