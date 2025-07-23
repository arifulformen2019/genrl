import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from datasets import VerificationMode, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# Assuming 'genrl.data.data_manager' is in the python path
from genrl.data.data_manager import TokenizedDataManager


class _TokenizedTextDataset(torch.utils.data.IterableDataset):
    """
    Internal IterableDataset that applies tokenization on the fly.
    """
    def __init__(self, dataset, tokenizer, name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.name = name

    def __iter__(self):
        for example in self.dataset:
            # This is where each piece of text gets turned into numbers (tokens)
            yield self.tokenizer(example[self.name])


class SerialHuggingFaceDataManager(TokenizedDataManager):
    """
    DataManager for loading and processing datasets from the Hugging Face Hub.
    
    OPTIMIZATION NOTE:
    This class is now optimized to automatically use the correct number of
    parallel processes for data loading based on the CPU it's running on.
    This prevents data pipeline bottlenecks and training stalls.
    """

    def __init__(
        self,
        path_or_name: str,
        tokenizer_path_or_name: str,
        batch_size: int,
        text_field_name: str = "text",
        access_token: str | None = None,
        # OPTIMIZATION: Set to `None` to enable auto-detection.
        # The code will automatically use the number of available CPU cores.
        num_workers: int | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initializes the DataManager.

        Args:
            path_or_name: Path or name of the dataset on the Hugging Face Hub.
            tokenizer_path_or_name: Path or name of the tokenizer.
            batch_size: The size of each data batch.
            text_field_name: The name of the column in the dataset containing the text.
            access_token: Hugging Face API token, if needed for private models.
            num_workers: Number of parallel processes for data loading. 
                         If None, it automatically uses all available CPU cores.
            tokenizer_kwargs: Additional arguments for the tokenizer.
        """
        self.path_or_name = path_or_name
        self.tokenizer_path_or_name = tokenizer_path_or_name
        self.batch_size = batch_size
        self.text_field_name = text_field_name
        self.access_token = access_token
        self.tokenizer_kwargs = tokenizer_kwargs
        self._train_iterator = None

        # OPTIMIZATION: Automatically detect and set the number of workers
        # if the user doesn't specify one. This makes the code efficient
        # on any machine without hardcoding assumptions.
        if num_workers is None:
            # os.cpu_count() can sometimes return None, so we fall back to 1.
            self.num_workers = os.cpu_count() or 1
        else:
            self.num_workers = num_workers
        
        print(f"INFO: Using {self.num_workers} parallel workers for data loading.")


    def initialize(self):
        """Initializes the tokenizer and the training data iterator."""
        _ = self.tokenizer
        self._train_iterator = iter(self.train_data_loader)

    def get_data_loader(self, path_or_name, partition):
        """Creates a DataLoader for a given dataset partition."""
        dataset = load_dataset(
            path_or_name,
            streaming=True,
            split=partition,
            verification_mode=VerificationMode.NO_CHECKS,
        )
        
        tokenized_dataset = _TokenizedTextDataset(
            dataset,
            self.tokenizer,
            self.text_field_name,
        )
        
        return torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            pin_memory=torch.cuda.is_available(),
        )

    @property
    def train_data_loader(self):
        """Property to get the training data loader."""
        return self.get_data_loader(self.path_or_name, "train")

    @property
    def tokenizer(self):
        """Property to initialize and get the tokenizer."""
        return AutoTokenizer.from_pretrained(
            self.tokenizer_path_or_name,
            token=self.access_token,
            padding_side="left",
            **(self.tokenizer_kwargs or {}),
        )

    def encode(self, text: str) -> Any:
        """Encodes a string of text into token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: Any) -> str:
        """Decodes a sequence of token IDs back into a string."""
        return self.tokenizer.decode(tokens)

    def get_round_data(self) -> Any:
        """Returns a single batch of training data."""
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_data_loader)
        try:
            batch = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_data_loader)
            batch = next(self._train_iterator)
        return batch

    def get_eval_data(self, name: str | None = None) -> Iterable[dict[str, Any]]:
        """Returns an iterable to the evaluation data."""
        name = name or "validation"
        return self.get_data_loader(self.path_or_name, name)
    
    def prepare_input(self, inputs: Dict[Any, List[List[Tuple[Any]]]]) -> Tuple[Any, Dict[int, Tuple[int, int, int]]]:
        raise NotImplementedError(
            "`prepare_input` is not implemented in `SerialHuggingFaceDataManager`. "
            "This DataManager is likely not compatible with the GRPOLanguageTrainerModule."
        )

    def prepare_actions(self, outputs: Any, index_mapping: Dict[int, Tuple[Any]]) -> Dict[Any, List[List[Any]]]:
        raise NotImplementedError("`prepare_actions` is not implemented in `SerialHuggingFaceDataManager`.")

    def prepare_states(self, swarm_states: Any) -> Dict[Any, List[List[Tuple[Any]]]]:
        raise NotImplementedError("`prepare_states` is not implemented in `SerialHuggingFaceDataManager`.")
