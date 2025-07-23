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
            yield self.tokenizer(example[self.name])


class SerialHuggingFaceDataManager(TokenizedDataManager):
    """
    DataManager for loading and processing datasets from the Hugging Face Hub.
    """

    def __init__(
        self,
        path_or_name: str,
        tokenizer_path_or_name: str,
        batch_size: int,
        text_field_name: str = "text",
        access_token: str | None = None,
        num_workers: int | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initializes the DataManager.
        """
        self.path_or_name = path_or_name
        self.tokenizer_path_or_name = tokenizer_path_or_name
        self.batch_size = batch_size
        self.text_field_name = text_field_name
        self.access_token = access_token
        self.tokenizer_kwargs = tokenizer_kwargs
        self._train_iterator = None

        # ======================================================================
        # DEBUGGING STEP 1: Forcing num_workers to 0
        # ----------------------------------------------------------------------
        # This change disables parallel data loading to test if the stall is
        # caused by a multiprocessing deadlock.
        #
        # - If the script RUNS after this change (even if slow), the problem is
        #   the data loader.
        # - If the script STILL STALLS, the problem is elsewhere.
        # ======================================================================
        self.num_workers = 0
        print("="*80)
        print("DEBUGGING: Forcing `num_workers = 0` to check for data loader deadlocks.")
        print("="*80)


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
