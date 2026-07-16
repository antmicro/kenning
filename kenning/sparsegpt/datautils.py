# Copyright (c) 2023-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Functionality to fetch calibration datasets.
"""

import logging
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_c4(
    n_samples: int,
    tokenizer: AutoTokenizer,
    seqlen: int = 4096,
    seed_constant: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    """
    Returns a calibration dataset that uses c4 dataset.
    https://huggingface.co/datasets/c4.

    Parameters
    ----------
    n_samples : int
        Number of samples in the calibration dataset
    tokenizer : AutoTokenizer
        Tokenizer that is used by the model
    seqlen : int
        Length of the sequence that is used by the model
    seed_constant : int
        This value determines how many times the set from which the samples
        are drawn is larger than the calibration dataset.
        The larger the value, the more random the samples are.
        It is introduced so that the dataset may be streamed from the disk,
        without loading the whole dataset into the memory.

    Returns
    -------
    List[Dict[str, torch.Tensor]]
        List of samples that constitute the calibration dataset
    """
    logger = logging.getLogger()
    verbosity = logger.level
    logger.setLevel(logging.ERROR)

    target_n_tokens = seqlen * n_samples * seed_constant
    tokenized_input_ids_list = []
    sep_tokens = tokenizer(" ")["input_ids"]

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )

    tokenized_input_ids = None
    # Batch load so that there is no network request for each
    # document
    for batch in dataset.iter(batch_size=1000):
        batch_tokens = tokenizer(batch["text"])["input_ids"]

        reached_target = False

        for sample_tokens in batch_tokens:
            tokenized_input_ids_list.extend(sample_tokens)
            if len(tokenized_input_ids_list) >= target_n_tokens:
                reached_target = True
                break
            tokenized_input_ids_list.extend(sep_tokens)
        if reached_target:
            break

    tokenized_input_ids = torch.tensor(
        tokenized_input_ids_list, dtype=torch.int64
    ).unsqueeze(0)

    samples = []
    for _ in range(n_samples):
        sample_idx = random.randint(
            0, tokenized_input_ids.shape[1] - seqlen - 1
        )
        sample = {
            "input_ids": tokenized_input_ids[
                :, sample_idx : sample_idx + seqlen
            ],
            "attention_mask": torch.ones(seqlen, dtype=torch.int64).unsqueeze(
                0
            ),
        }
        samples.append(sample)

    logger.setLevel(verbosity)
    return samples
