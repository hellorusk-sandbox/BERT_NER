import logging
import os

from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments
)

import numpy as np
from datasets import load_dataset



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    cache_dir: Optional[str] = field(
        default="model",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArugments:
    dataset_name: str


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArugments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)


    if data_args.dataset_name is not None:
        # Download a dataset using Huggingface/datasets library
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        # Set up your own dataset
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file 
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file 
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file 
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    if training_args.do_train:
        column_names = datasets["train"].column_names 
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names 
        features = datasets["validation"].features  
    text_column_name = column_names[0]
    label_column_name = column_names[1]
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    num_labels = len(label_list)


    # Load model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config
    )


    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            # The option below is required if the sentences are already split into words.
            is_split_into_words=True,
        )
        labels = []