# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


import logging
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_c4(
    n_samples: str,
    tokenizer: AutoTokenizer,
    seqlen: int = 4096,
    seed_constant: int = 5,
) -> List[Dict[str, torch.Tensor]]:
    """
    Returns a calibration dataset that uses c4 dataset.
    https://huggingface.co/datasets/c4.

    Parameters
    ----------
    n_samples : str
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

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    tokenized_input_ids = None
    for sample in dataset:
        tokenized_sample_input_ids = tokenizer(
            sample["text"], return_tensors="pt"
        ).input_ids

        if tokenized_input_ids is None:
            tokenized_input_ids = tokenized_sample_input_ids
        else:
            tokenized_input_ids = torch.cat(
                (tokenized_input_ids, tokenized_sample_input_ids), dim=1
            )

        if tokenized_input_ids.shape[1] >= seqlen * n_samples * seed_constant:
            break

        # Appending a whitespace token to the end of the input_ids
        # to separate the samples. This is needed because the tokenizer
        # does not add a whitespace token at the end of the input_ids
        tokenized_whitespace_input_ids = tokenizer(
            " ", return_tensors="pt"
        ).input_ids
        tokenized_input_ids = torch.cat(
            (tokenized_input_ids, tokenized_whitespace_input_ids), dim=1
        )

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
