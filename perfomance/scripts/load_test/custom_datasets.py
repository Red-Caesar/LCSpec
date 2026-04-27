import abc
import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZED_DIR = Path(os.path.join(CURRENT_DIR, "tokenized_datasets"))
PROMPT_TYPE = Union[List[str], List[List[int]]]


class BaseDataset(abc.ABC):
    def __init__(self, tokenized_dataset_name: str = "") -> None:
        self.tokenized_dataset_name = tokenized_dataset_name

    @abc.abstractmethod
    def prepare_prompts(self) -> List[str]:
        """Get prompts from the dataset."""
        raise NotImplementedError("Subclasses should implement this method.")

    def tokenize_prompts(
        self, prompts: List[str], tokenizer: AutoTokenizer
    ) -> List[List[int]]:
        assert tokenizer is not None

        tokenizer_name = tokenizer.name_or_path.split("/")[-1]
        path_to_tokenized_datasets = DEFAULT_TOKENIZED_DIR.joinpath(tokenizer_name)
        os.makedirs(path_to_tokenized_datasets, exist_ok=True)
        path_to_dataset = path_to_tokenized_datasets.joinpath(
            self.tokenized_dataset_name
        )

        if os.path.exists(path_to_dataset):
            print("Loading tokenized dataset from disk...")
            with open(path_to_dataset, "rb") as f:
                tokenized_prompts = pickle.load(f)
        else:
            print("Starting tokenization...")
            tokenized_prompts = [
                tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
            ]
            print("Tokenization is over")
            with open(path_to_dataset, "wb") as f:
                pickle.dump(tokenized_prompts, f)

        return tokenized_prompts

    def get_prompts(self, tokenizer: Optional[AutoTokenizer] = None) -> PROMPT_TYPE:
        """Get prompts from the dataset."""
        prompts = self.prepare_prompts()
        if tokenizer:
            prompts = self.tokenize_prompts(prompts, tokenizer)  # type: ignore

        return prompts


class CodeDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(
            tokenized_dataset_name="code_tokenized.pickle",
        )

    def load_data(self):
        try:
            self.data = load_dataset("google-research-datasets/mbpp", "full")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    def prepare_prompts(self) -> List[str]:
        self.load_data()
        return list(self.data["test"]["text"])


class ShareGPT4Dataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(
            tokenized_dataset_name="sharegpt4_tokenized.pickle",
        )

    def load_data(self):
        try:
            self.data = load_dataset("shibing624/sharegpt_gpt4", split="train")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    def prepare_prompts(self) -> List[str]:
        self.load_data()
        prompts = [conv[0]["value"] for conv in self.data["conversations"]]
        return prompts


class SummaryDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(
            tokenized_dataset_name="summary_tokenized.pickle",
        )

    def load_data(self):
        try:
            self.data = load_dataset(
                "ChicagoHAI/CaseSumm", split="train", trust_remote_code=True
            )
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    def prepare_prompts(self) -> List[str]:
        self.load_data()
        system_prompt = "## TASK: Make a summary of the following text:\n\n ## TEXT: "
        prompts = [system_prompt + doc for doc in self.data["opinion"]]
        return prompts


class DatasetFactory:
    def __init__(
        self,
        dataset_type: str,
        input_tokens_distribution: str,
        tokenizer_name: Optional[str] = None,
    ) -> None:
        if dataset_type == "code":
            self.dataset = CodeDataset()
        elif dataset_type == "sharegpt4":
            self.dataset = ShareGPT4Dataset()
        elif dataset_type == "summary":
            self.dataset = SummaryDataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        tokenizer = None
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        prompts = self.dataset.get_prompts(tokenizer)
        distribution, parameters = input_tokens_distribution[:-1].split("(")
        print("Starting prompt truncation...")
        if distribution == "const":
            context_len = int(parameters)
            self.prompts = DatasetFactory.const_distribution(prompts, context_len)
        elif distribution == "normal":
            mean, std = list(map(int, parameters.split(",")))
            self.prompts = DatasetFactory.normal_distribution(prompts, mean, std)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        if tokenizer:
            self.prompts = tokenizer.batch_decode(self.prompts)
        print("Ended prompt truncation...")

    @staticmethod
    def const_distribution(prompts: PROMPT_TYPE, context_len: int) -> PROMPT_TYPE:
        truncated_prompts = []
        for prompt in prompts:
            if len(prompt) < context_len:
                continue
            truncated_prompts.append(prompt[:context_len])
        return truncated_prompts

    @staticmethod
    def normal_distribution(prompts: PROMPT_TYPE, mean: int, std: int) -> PROMPT_TYPE:
        truncated_prompts = []
        for prompt in prompts:
            context_len = int(np.random.normal(mean, std))
            if context_len <= 0 or len(prompt) < context_len:
                continue
            truncated_prompts.append(prompt[:context_len])
        return truncated_prompts
