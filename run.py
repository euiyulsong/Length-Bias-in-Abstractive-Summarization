#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summary_drift_qwen3.py

======================================================================
PURPOSE
======================================================================

Question:
    If Qwen3 is fine-tuned on 1,000 genuinely instruction-formatted,
    high-quality summarization examples, does recursive summarization
    still get longer over repeated updates?

TRAIN DATA
----------
1) HuggingFaceH4/no_robots
   - category == "Summarize"
   - ~420 examples
   - human-created
   - ORIGINAL messages:
       system / user / assistant
   - no prompt rewriting
   - no synthetic target generation

2) Open-Orca/OpenOrca
   - fill remaining examples until total = 1,000
   - ORIGINAL:
       system_prompt / question / response
   - filter only real summarization instructions
   - no prompt rewriting
   - target is existing dataset response

MODEL
-----
Qwen/Qwen3-1.7B

TRAIN
-----
LoRA SFT
1 epoch
assistant-only loss

EVAL
----
ONLY Hugging Face transformers model.generate()
NO vLLM
NO OpenAI API
NO teacher model
NO LLM judge

Two evaluations:

A. held-out summarization
   PRETRAINED vs SFT
   - ROUGE
   - generated length
   - reference length

B. recursive length-drift stress test

   S1 = summarize(chunk1)

   S2 = summarize(
       previous_summary=S1,
       new_information=chunk2
   )

   ...

Metrics:
    growth_ratio
    growing_step_rate
    final_to_first_ratio
    token_slope
    duplicate_ratio

======================================================================
INSTALL
======================================================================

pip install -U \
    "transformers>=4.51.0" \
    datasets \
    accelerate \
    peft \
    evaluate \
    rouge_score \
    pandas \
    matplotlib \
    tqdm

======================================================================
RUN
======================================================================

python3 summary_drift_qwen3.py all

Or:

python3 summary_drift_qwen3.py prepare

python3 summary_drift_qwen3.py train

python3 summary_drift_qwen3.py eval-quality

python3 summary_drift_qwen3.py eval-drift

"""

import os
import re
import gc
import json
import math
import random
import argparse
import statistics

from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm


# ======================================================================
# CONFIG
# ======================================================================

SEED = 42

MODEL_NAME = "Qwen/Qwen3-1.7B"

OUT_DIR = "./qwen3_summary_drift"

TRAIN_N = 1000

NO_ROBOTS_DATASET = "HuggingFaceH4/no_robots"
OPENORCA_DATASET = "Open-Orca/OpenOrca"

MAX_LENGTH = 2048

EPOCHS = 1.0

BATCH_SIZE = 8
GRAD_ACCUM = 2

LR = 2e-4

MAX_NEW_TOKENS = 256

QUALITY_EVAL_N = 100

DRIFT_CHAINS = 100
DRIFT_STEPS = 8


# ======================================================================
# UTILS
# ======================================================================

def seed_everything(seed=42):

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass


def mkdir(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def save_json(obj, path):

    path = Path(path)

    mkdir(path.parent)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def normalize_space(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def sentence_split(text):

    text = str(text).strip()

    if not text:
        return []

    parts = re.split(
        r"(?:\n+|(?<=[.!?])\s+)",
        text
    )

    return [
        normalize_space(x).lower()
        for x in parts
        if normalize_space(x)
    ]


def duplicate_ratio(text):

    sents = sentence_split(text)

    if len(sents) <= 1:
        return 0.0

    counts = Counter(sents)

    duplicates = sum(
        max(0, n - 1)
        for n in counts.values()
    )

    return duplicates / len(sents)


def slope(values):

    if len(values) < 2:
        return 0.0

    xs = list(
        range(1, len(values) + 1)
    )

    mx = statistics.mean(xs)
    my = statistics.mean(values)

    numerator = sum(
        (x - mx) * (y - my)
        for x, y in zip(xs, values)
    )

    denominator = sum(
        (x - mx) ** 2
        for x in xs
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ======================================================================
# DETECT SUMMARIZATION PROMPTS
# ======================================================================

SUMMARY_PATTERNS = [

    r"\bsummarize\b",
    r"\bsummarise\b",

    r"\bsummarization\b",
    r"\bsummarisation\b",

    r"\bsummary\b",

    r"\btl\s*;\s*dr\b",

    r"\bbrief summary\b",

    r"\bshort summary\b",

    r"\bprovide a concise overview\b",

    r"\bprovide an overview of\b",

    r"\bwrite a summary\b",

    r"\bgive a summary\b",

    r"\bcreate a summary\b",
]


def is_summary_instruction(text):

    x = str(text).lower()

    return any(
        re.search(pattern, x)
        for pattern in SUMMARY_PATTERNS
    )


# ======================================================================
# MESSAGE HELPERS
# ======================================================================

def valid_messages(messages):

    if not isinstance(messages, list):
        return False

    roles = [
        x.get("role")
        for x in messages
        if isinstance(x, dict)
    ]

    return (
        "user" in roles
        and
        "assistant" in roles
    )


def last_assistant(messages):

    for msg in reversed(messages):

        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()

    return ""


def all_user_text(messages):

    return "\n\n".join(
        x.get("content", "")
        for x in messages
        if x.get("role") == "user"
    )


# ======================================================================
# NO ROBOTS
# ======================================================================

def load_no_robots():

    from datasets import load_dataset

    print()
    print("=" * 100)
    print("NO ROBOTS: HUMAN SUMMARIZATION DATA")
    print("=" * 100)

    ds = load_dataset(
        NO_ROBOTS_DATASET
    )

    print(ds)

    # HF renamed splits over time in different mirrors.
    if "train" in ds:
        train = ds["train"]

    elif "train_sft" in ds:
        train = ds["train_sft"]

    else:
        raise RuntimeError(
            f"Unexpected No Robots splits: {ds.keys()}"
        )

    examples = []

    for row in train:

        category = str(
            row.get("category", "")
        )

        if category.lower() != "summarize":
            continue

        messages = row["messages"]

        if not valid_messages(messages):
            continue

        assistant = last_assistant(messages)

        if not assistant:
            continue

        examples.append({

            "source":
                "no_robots",

            "source_id":
                row.get(
                    "prompt_id",
                    ""
                ),

            # IMPORTANT:
            # unmodified original messages
            "messages":
                messages,

            "prompt":
                row.get(
                    "prompt",
                    ""
                ),

            "reference":
                assistant,
        })

    print(
        "No Robots summarization examples:",
        len(examples)
    )

    return examples


# ======================================================================
# OPEN ORCA
# ======================================================================

def load_openorca_summary(
    needed,
    seed=42,
    max_scan=500000,
):

    """
    OpenOrca is huge, so stream it.

    We only select rows where the ORIGINAL question itself
    is clearly a summarization instruction.

    Original fields:
        system_prompt
        question
        response

    We do NOT generate or rewrite any of them.
    """

    from datasets import load_dataset

    print()
    print("=" * 100)
    print("OPENORCA: SUMMARIZATION INSTRUCTION FILTER")
    print("=" * 100)

    ds = load_dataset(
        OPENORCA_DATASET,
        split="train",
        streaming=True,
    )

    # deterministic shuffle buffer
    ds = ds.shuffle(
        seed=seed,
        buffer_size=10000
    )

    output = []

    scanned = 0

    progress = tqdm(
        total=needed,
        desc="OpenOrca summaries"
    )

    for row in ds:

        scanned += 1

        if scanned > max_scan:
            break

        question = str(
            row.get(
                "question",
                ""
            )
        ).strip()

        response = str(
            row.get(
                "response",
                ""
            )
        ).strip()

        system = str(
            row.get(
                "system_prompt",
                ""
            )
        ).strip()

        if not question:
            continue

        if not response:
            continue

        if not system:
            continue

        if not is_summary_instruction(
            question
        ):
            continue

        # Avoid CoT-oriented system prompts because
        # we want direct summaries, not reasoning traces.
        system_lower = system.lower()

        banned_system_terms = [
            "step by step",
            "chain of thought",
            "explain your reasoning",
            "reason step-by-step",
        ]

        if any(
            term in system_lower
            for term in banned_system_terms
        ):
            continue

        messages = [

            {
                "role": "system",
                "content": system,
            },

            {
                "role": "user",
                "content": question,
            },

            {
                "role": "assistant",
                "content": response,
            },
        ]

        output.append({

            "source":
                "openorca",

            "source_id":
                row.get(
                    "id",
                    ""
                ),

            "messages":
                messages,

            "prompt":
                question,

            "reference":
                response,
        })

        progress.update(1)

        if len(output) >= needed:
            break

    progress.close()

    print(
        f"Scanned {scanned:,} OpenOrca rows."
    )

    print(
        f"Selected {len(output):,} "
        f"summarization rows."
    )

    if len(output) < needed:

        raise RuntimeError(
            f"Could only find {len(output)} "
            f"OpenOrca summarization examples; "
            f"needed {needed}."
        )

    return output


# ======================================================================
# PREPARE TRAIN DATA
# ======================================================================

def prepare(args):

    seed_everything(args.seed)

    out_dir = Path(args.out_dir)

    mkdir(out_dir)

    print("=" * 100)
    print("BUILD 1,000 ORIGINAL INSTRUCTION-SUMMARY EXAMPLES")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Human examples first
    # ------------------------------------------------------------------

    no_robots = load_no_robots()

    random.Random(
        args.seed
    ).shuffle(
        no_robots
    )

    # Take all human summary examples.
    selected_human = no_robots[
        :min(
            len(no_robots),
            args.train_n
        )
    ]

    remaining = (
        args.train_n
        - len(selected_human)
    )

    # ------------------------------------------------------------------
    # Fill to 1000 with original OpenOrca summary instructions
    # ------------------------------------------------------------------

    openorca = []

    if remaining > 0:

        openorca = load_openorca_summary(
            needed=remaining,
            seed=args.seed,
        )

    train = (
        selected_human
        +
        openorca
    )

    random.Random(
        args.seed
    ).shuffle(train)

    train = train[
        :args.train_n
    ]

    save_json(
        train,
        out_dir
        /
        "train_1000.json"
    )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    source_counts = Counter(
        x["source"]
        for x in train
    )

    print()
    print("=" * 100)
    print("TRAIN DATA COMPOSITION")
    print("=" * 100)

    print(
        "total:",
        len(train)
    )

    for source, count in source_counts.items():

        print(
            f"{source:20s}: {count}"
        )

    print()
    print("FIRST EXAMPLE")
    print(
        json.dumps(
            train[0],
            indent=2,
            ensure_ascii=False,
        )[:5000]
    )

    # Verify role distribution.
    role_counts = Counter()

    for x in train:

        for m in x["messages"]:

            role_counts[
                m["role"]
            ] += 1

    print()
    print(
        "message role counts:",
        role_counts
    )


# ======================================================================
# ASSISTANT-ONLY SFT DATASET
# ======================================================================

class AssistantOnlyDataset:

    def __init__(
        self,
        examples,
        tokenizer,
        max_length,
    ):

        import torch

        self.items = []

        for example in tqdm(
            examples,
            desc="Tokenizing SFT",
        ):

            messages = example[
                "messages"
            ]

            if (
                not messages
                or
                messages[-1]["role"]
                != "assistant"
            ):
                continue

            # Everything except final answer.
            prompt_messages = messages[:-1]

            assistant_text = messages[-1][
                "content"
            ]

            # ----------------------------------------------------------
            # Qwen chat prompt
            # ----------------------------------------------------------

            prompt = tokenizer.apply_chat_template(

                prompt_messages,

                tokenize=False,

                add_generation_prompt=True,

                enable_thinking=False,
            )

            # We append only assistant text + EOS.
            #
            # This avoids accidentally putting loss on system/user
            # tokens and makes boundary handling transparent.

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            target_ids = tokenizer(
                assistant_text
                +
                tokenizer.eos_token,

                add_special_tokens=False,
            )["input_ids"]

            # ----------------------------------------------------------
            # Truncation
            #
            # Preserve response completely where possible.
            # Trim prompt from the LEFT if context is too long.
            # ----------------------------------------------------------

            if len(target_ids) >= max_length:

                target_ids = target_ids[
                    :max_length
                ]

                prompt_ids = []

            else:

                allowed_prompt = (
                    max_length
                    -
                    len(target_ids)
                )

                if len(prompt_ids) > allowed_prompt:

                    prompt_ids = prompt_ids[
                        -allowed_prompt:
                    ]

            input_ids = (
                prompt_ids
                +
                target_ids
            )

            labels = (
                [-100] * len(prompt_ids)
                +
                target_ids
            )

            self.items.append({

                "input_ids":
                    torch.tensor(
                        input_ids,
                        dtype=torch.long,
                    ),

                "attention_mask":
                    torch.ones(
                        len(input_ids),
                        dtype=torch.long,
                    ),

                "labels":
                    torch.tensor(
                        labels,
                        dtype=torch.long,
                    ),
            })

    def __len__(self):

        return len(
            self.items
        )

    def __getitem__(self, idx):

        return self.items[
            idx
        ]


class Collator:

    def __init__(
        self,
        tokenizer
    ):

        self.tokenizer = tokenizer

    def __call__(
        self,
        examples
    ):

        import torch

        max_len = max(
            len(x["input_ids"])
            for x in examples
        )

        ids = []
        masks = []
        labels = []

        for x in examples:

            n = len(
                x["input_ids"]
            )

            pad = max_len - n

            ids.append(

                torch.cat([

                    x["input_ids"],

                    torch.full(
                        (pad,),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ])
            )

            masks.append(

                torch.cat([

                    x["attention_mask"],

                    torch.zeros(
                        pad,
                        dtype=torch.long,
                    ),
                ])
            )

            labels.append(

                torch.cat([

                    x["labels"],

                    torch.full(
                        (pad,),
                        -100,
                        dtype=torch.long,
                    ),
                ])
            )

        return {

            "input_ids":
                torch.stack(ids),

            "attention_mask":
                torch.stack(masks),

            "labels":
                torch.stack(labels),
        }


# ======================================================================
# TRAIN
# ======================================================================

def train(args):

    import torch

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
    )

    from peft import (
        LoraConfig,
        get_peft_model,
    )

    seed_everything(
        args.seed
    )

    out_dir = Path(
        args.out_dir
    )

    train_path = (
        out_dir
        /
        "train_1000.json"
    )

    if not train_path.exists():

        prepare(args)

    examples = load_json(
        train_path
    )

    examples = examples[
        :args.train_n
    ]

    print()
    print("=" * 100)
    print("TRAIN")
    print("=" * 100)

    print(
        "model:",
        args.model
    )

    print(
        "examples:",
        len(examples)
    )

    print(
        "epochs:",
        args.epochs
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    if (
        torch.cuda.is_available()
        and
        torch.cuda.is_bf16_supported()
    ):

        dtype = torch.bfloat16

    elif torch.cuda.is_available():

        dtype = torch.float16

    else:

        dtype = torch.float32

    model = (
        AutoModelForCausalLM
        .from_pretrained(

            args.model,

            torch_dtype=dtype,

            trust_remote_code=True,
        )
    )

    if args.gradient_checkpointing:

        model.gradient_checkpointing_enable()

        model.enable_input_require_grads()

        model.config.use_cache = False

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    config = LoraConfig(

        task_type="CAUSAL_LM",

        r=args.lora_r,

        lora_alpha=
            args.lora_alpha,

        lora_dropout=0.05,

        bias="none",

        target_modules=[

            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",

            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        config
    )

    model.print_trainable_parameters()

    dataset = AssistantOnlyDataset(

        examples=examples,

        tokenizer=tokenizer,

        max_length=
            args.max_length,
    )

    output_path = (
        out_dir
        /
        "qwen3_summary_lora"
    )

    train_args = TrainingArguments(

        output_dir=
            str(output_path),

        # USER REQUIREMENT
        num_train_epochs=
            args.epochs,

        per_device_train_batch_size=
            args.batch_size,

        gradient_accumulation_steps=
            args.grad_accum,

        learning_rate=
            args.lr,

        warmup_ratio=
            0.03,

        weight_decay=
            0.01,

        lr_scheduler_type=
            "cosine",

        logging_steps=
            5,

        save_strategy=
            "epoch",

        save_total_limit=
            1,

        bf16=(
            torch.cuda.is_available()
            and
            torch.cuda.is_bf16_supported()
        ),

        fp16=(
            torch.cuda.is_available()
            and
            not torch.cuda.is_bf16_supported()
        ),

        gradient_checkpointing=
            args.gradient_checkpointing,

        optim=
            "adamw_torch",

        report_to=
            "none",

        remove_unused_columns=
            False,

        dataloader_num_workers=
            2,

        seed=
            args.seed,
    )

    trainer = Trainer(

        model=model,

        args=train_args,

        train_dataset=
            dataset,

        data_collator=
            Collator(tokenizer),
    )

    trainer.train()

    model.save_pretrained(
        output_path
    )

    tokenizer.save_pretrained(
        output_path
    )

    print()
    print(
        "Saved adapter:",
        output_path
    )


# ======================================================================
# HF GENERATOR
# ======================================================================

class Generator:

    """
    IMPORTANT:

    The ONLY inference implementation used in this experiment is:

        transformers.AutoModelForCausalLM
        model.generate()

    No vLLM.
    """

    def __init__(
        self,
        base_model,
        adapter=None,
    ):

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )

        self.torch = torch

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                base_model,
                trust_remote_code=True,
            )
        )

        if (
            self.tokenizer.pad_token_id
            is None
        ):

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        if (
            torch.cuda.is_available()
            and
            torch.cuda.is_bf16_supported()
        ):

            dtype = torch.bfloat16

        elif torch.cuda.is_available():

            dtype = torch.float16

        else:

            dtype = torch.float32

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(

                base_model,

                torch_dtype=dtype,

                device_map="auto",

                trust_remote_code=True,
            )
        )

        if adapter:

            from peft import (
                PeftModel
            )

            self.model = (
                PeftModel
                .from_pretrained(
                    self.model,
                    adapter,
                )
            )

        self.model.eval()

    def ntokens(
        self,
        text
    ):

        return len(
            self.tokenizer(
                str(text),
                add_special_tokens=False,
            )[
                "input_ids"
            ]
        )

    def generate(
        self,
        messages,
        max_new_tokens=256,
    ):

        torch = self.torch

        inputs = (
            self.tokenizer
            .apply_chat_template(

                messages,

                tokenize=True,

                add_generation_prompt=True,

                enable_thinking=False,

                return_dict=True,

                return_tensors="pt",
            )
        )

        device = next(
            self.model.parameters()
        ).device

        inputs = {

            k: v.to(device)

            for k, v
            in inputs.items()
        }

        input_len = (
            inputs["input_ids"]
            .shape[1]
        )

        # --------------------------------------------------------------
        # Controlled experiment:
        #
        # deterministic decoding avoids sampling noise in length.
        # --------------------------------------------------------------

        with torch.inference_mode():

            output = (
                self.model.generate(

                    **inputs,

                    max_new_tokens=
                        max_new_tokens,

                    do_sample=False,

                    use_cache=True,

                    pad_token_id=
                        self.tokenizer.pad_token_id,

                    eos_token_id=
                        self.tokenizer.eos_token_id,
                )
            )

        generated = output[
            0,
            input_len:
        ]

        return (
            self.tokenizer
            .decode(

                generated,

                skip_special_tokens=True,
            )
            .strip()
        )


# ======================================================================
# HELD-OUT NO ROBOTS QUALITY DATA
# ======================================================================

def load_no_robots_test():

    from datasets import (
        load_dataset
    )

    ds = load_dataset(
        NO_ROBOTS_DATASET
    )

    if "test" in ds:

        test = ds["test"]

    elif "test_sft" in ds:

        test = ds[
            "test_sft"
        ]

    else:

        raise RuntimeError(
            "No No-Robots test split."
        )

    examples = []

    for row in test:

        if str(
            row.get(
                "category",
                ""
            )
        ).lower() != "summarize":

            continue

        messages = row[
            "messages"
        ]

        if not valid_messages(
            messages
        ):
            continue

        if (
            messages[-1][
                "role"
            ]
            != "assistant"
        ):
            continue

        examples.append({

            "messages":
                messages,

            "reference":
                messages[-1][
                    "content"
                ],
        })

    return examples


# ======================================================================
# QUALITY EVAL
# ======================================================================

def run_quality(
    name,
    generator,
    examples,
    max_new_tokens,
):

    predictions = []
    references = []

    rows = []

    for example in tqdm(
        examples,
        desc=name,
    ):

        prompt_messages = (
            example[
                "messages"
            ][:-1]
        )

        reference = (
            example[
                "reference"
            ]
        )

        prediction = (
            generator.generate(
                prompt_messages,
                max_new_tokens,
            )
        )

        predictions.append(
            prediction
        )

        references.append(
            reference
        )

        rows.append({

            "model":
                name,

            "prediction":
                prediction,

            "reference":
                reference,

            "pred_tokens":
                generator.ntokens(
                    prediction
                ),

            "ref_tokens":
                generator.ntokens(
                    reference
                ),
        })

    import evaluate

    rouge = evaluate.load(
        "rouge"
    )

    scores = rouge.compute(

        predictions=
            predictions,

        references=
            references,

        use_stemmer=True,
    )

    scores[
        "model"
    ] = name

    scores[
        "n"
    ] = len(rows)

    scores[
        "mean_pred_tokens"
    ] = statistics.mean(
        x["pred_tokens"]
        for x in rows
    )

    scores[
        "mean_ref_tokens"
    ] = statistics.mean(
        x["ref_tokens"]
        for x in rows
    )

    return rows, scores


def eval_quality(args):

    import torch

    examples = (
        load_no_robots_test()
    )

    examples = examples[
        :args.quality_eval_n
    ]

    print()
    print(
        "Held-out human summary examples:",
        len(examples)
    )

    all_rows = []
    summaries = []

    # --------------------------------------------------------------
    # Base
    # --------------------------------------------------------------

    gen = Generator(
        args.model
    )

    rows, score = run_quality(

        "PRETRAINED",

        gen,

        examples,

        args.max_new_tokens,
    )

    all_rows += rows
    summaries.append(score)

    del gen

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # SFT
    # --------------------------------------------------------------

    adapter = (
        Path(args.out_dir)
        /
        "qwen3_summary_lora"
    )

    gen = Generator(

        args.model,

        str(adapter),
    )

    rows, score = run_quality(

        "SFT",

        gen,

        examples,

        args.max_new_tokens,
    )

    all_rows += rows
    summaries.append(score)

    del gen

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    detail = pd.DataFrame(
        all_rows
    )

    summary = pd.DataFrame(
        summaries
    )

    detail.to_csv(

        Path(args.out_dir)
        /
        "quality_detail.csv",

        index=False,
    )

    summary.to_csv(

        Path(args.out_dir)
        /
        "quality_summary.csv",

        index=False,
    )

    print()
    print("=" * 100)
    print("QUALITY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            float_format=
                lambda x: f"{x:.4f}",
        )
    )


# ======================================================================
# EXTRACT SOURCE TEXT FROM SUMMARY PROMPT
# ======================================================================

def extract_summary_document(
    user_prompt
):

    """
    No Robots summary prompts use many natural forms.

    We DON'T need perfect source extraction for training.

    For drift evaluation we want chunks of original source text,
    so this strips common leading summarization instructions.
    """

    text = str(
        user_prompt
    ).strip()

    patterns = [

        r"^please\s+summari[sz]e[^:\n]*:\s*",

        r"^summari[sz]e[^:\n]*:\s*",

        r"^write\s+(?:a\s+)?summary[^:\n]*:\s*",

        r"^give\s+(?:me\s+)?(?:a\s+)?summary[^:\n]*:\s*",

        r"^provide\s+(?:a\s+)?(?:brief|concise)?\s*summary[^:\n]*:\s*",
    ]

    result = text

    for pattern in patterns:

        candidate = re.sub(
            pattern,
            "",
            text,
            flags=
                re.I | re.S,
        )

        if candidate != text:

            result = candidate
            break

    return result.strip()


# ======================================================================
# BUILD RECURSIVE STRESS TEST
# ======================================================================

RECURSIVE_SYSTEM = """
You are a summarization assistant.

Maintain a concise and faithful summary as new information arrives.

Preserve important information from the previous summary and the new text.
Merge duplicate or overlapping information.
Do not invent facts.
Return only the updated summary.
""".strip()


def make_first_messages(
    chunk
):

    return [

        {
            "role":
                "system",

            "content":
                RECURSIVE_SYSTEM,
        },

        {
            "role":
                "user",

            "content":
                f"""Summarize the following text concisely.

Text:
{chunk}

Summary:""",
        },
    ]


def make_update_messages(
    previous,
    chunk,
):

    return [

        {
            "role":
                "system",

            "content":
                RECURSIVE_SYSTEM,
        },

        {
            "role":
                "user",

            "content":
                f"""Update the existing summary with the new information.

Existing summary:
{previous}

New information:
{chunk}

Return one concise updated summary:""",
        },
    ]


def build_drift_chains(
    args
):

    """
    Create chains from REAL summarization documents.

    Each held-out document is split into textual chunks.

    This is much cleaner than feeding unrelated conversations:
    every recursive step adds more text from the SAME original document.

    The model is therefore doing:

        prefix summary -> larger prefix summary
    """

    examples = (
        load_no_robots_test()
    )

    documents = []

    for example in examples:

        messages = example[
            "messages"
        ]

        users = [
            x["content"]
            for x in messages
            if x["role"] == "user"
        ]

        if not users:
            continue

        document = (
            extract_summary_document(
                users[-1]
            )
        )

        # Ignore tiny prompts.
        words = document.split()

        if len(words) < (
            args.drift_steps * 30
        ):
            continue

        documents.append(
            document
        )

    # No Robots test may have too few long summarize examples.
    #
    # Fill the stress set using fresh OpenOrca summaries,
    # but these are NOT used as references here.
    #
    # They provide actual summarization source prompts.

    if len(documents) < args.drift_chains:

        need = (
            args.drift_chains
            -
            len(documents)
        )

        extra = load_openorca_summary(
            needed=need + 20,
            seed=args.seed + 1000,
        )

        for x in extra:

            users = [
                m["content"]
                for m in x["messages"]
                if m["role"] == "user"
            ]

            if not users:
                continue

            doc = extract_summary_document(
                users[-1]
            )

            if len(doc.split()) >= (
                args.drift_steps * 30
            ):

                documents.append(
                    doc
                )

            if len(documents) >= (
                args.drift_chains
            ):
                break

    documents = documents[
        :args.drift_chains
    ]

    chains = []

    for i, document in enumerate(
        documents
    ):

        words = (
            document.split()
        )

        # Equal-size incremental chunks.
        n_steps = min(
            args.drift_steps,
            max(
                2,
                len(words) // 30
            )
        )

        chunk_size = math.ceil(
            len(words)
            /
            n_steps
        )

        chunks = []

        for j in range(
            0,
            len(words),
            chunk_size,
        ):

            chunk = " ".join(
                words[
                    j:
                    j + chunk_size
                ]
            )

            if chunk.strip():

                chunks.append(
                    chunk
                )

        chunks = chunks[
            :n_steps
        ]

        if len(chunks) >= 2:

            chains.append({

                "id":
                    f"chain_{i}",

                "chunks":
                    chunks,
            })

    return chains


# ======================================================================
# DRIFT EVAL
# ======================================================================

def run_drift_model(
    name,
    generator,
    chains,
    args,
):

    rows = []

    for chain in tqdm(
        chains,
        desc=f"drift {name}",
    ):

        previous = ""

        total_source_tokens = 0

        for step, chunk in enumerate(
            chain["chunks"],
            start=1,
        ):

            chunk_tokens = (
                generator.ntokens(
                    chunk
                )
            )

            total_source_tokens += (
                chunk_tokens
            )

            prev_tokens = (
                generator.ntokens(
                    previous
                )
                if previous
                else 0
            )

            if step == 1:

                messages = (
                    make_first_messages(
                        chunk
                    )
                )

            else:

                messages = (
                    make_update_messages(
                        previous,
                        chunk,
                    )
                )

            summary = (
                generator.generate(

                    messages,

                    max_new_tokens=
                        args.max_new_tokens,
                )
            )

            new_tokens = (
                generator.ntokens(
                    summary
                )
            )

            if prev_tokens:

                growth = (
                    new_tokens
                    /
                    prev_tokens
                )

            else:

                growth = None

            rows.append({

                "model":
                    name,

                "chain_id":
                    chain["id"],

                "step":
                    step,

                "prev_tokens":
                    prev_tokens,

                "new_tokens":
                    new_tokens,

                "delta_tokens":
                    (
                        new_tokens
                        -
                        prev_tokens
                    ),

                "growth_ratio":
                    growth,

                "source_tokens_seen":
                    total_source_tokens,

                "compression_ratio":
                    (
                        new_tokens
                        /
                        total_source_tokens
                    ),

                "duplicate_ratio":
                    duplicate_ratio(
                        summary
                    ),

                "previous":
                    previous,

                "chunk":
                    chunk,

                "summary":
                    summary,
            })

            previous = summary

    return rows


def aggregate_drift(
    df
):

    conv_rows = []

    for (
        model,
        cid
    ), g in df.groupby([
        "model",
        "chain_id",
    ]):

        g = g.sort_values(
            "step"
        )

        lengths = (
            g["new_tokens"]
            .tolist()
        )

        ratios = (
            g.loc[
                g["step"] > 1,
                "growth_ratio"
            ]
            .dropna()
            .tolist()
        )

        conv_rows.append({

            "model":
                model,

            "chain_id":
                cid,

            "steps":
                len(lengths),

            "mean_growth_ratio":
                (
                    statistics.mean(
                        ratios
                    )
                    if ratios
                    else float("nan")
                ),

            "growing_step_rate":
                (
                    sum(
                        r > 1.0
                        for r in ratios
                    )
                    /
                    len(ratios)

                    if ratios
                    else float("nan")
                ),

            "final_to_first_ratio":
                (
                    lengths[-1]
                    /
                    lengths[0]

                    if lengths
                    and lengths[0]
                    else float("nan")
                ),

            "token_slope":
                slope(
                    lengths
                ),

            "first_tokens":
                lengths[0],

            "final_tokens":
                lengths[-1],

            "duplicate_ratio":
                g[
                    "duplicate_ratio"
                ].mean(),
        })

    conv = pd.DataFrame(
        conv_rows
    )

    summary = (

        conv
        .groupby(
            "model"
        )
        .agg(

            n=(
                "chain_id",
                "count",
            ),

            mean_growth_ratio=(
                "mean_growth_ratio",
                "mean",
            ),

            growing_step_rate=(
                "growing_step_rate",
                "mean",
            ),

            final_to_first_ratio=(
                "final_to_first_ratio",
                "mean",
            ),

            token_slope=(
                "token_slope",
                "mean",
            ),

            first_tokens=(
                "first_tokens",
                "mean",
            ),

            final_tokens=(
                "final_tokens",
                "mean",
            ),

            duplicate_ratio=(
                "duplicate_ratio",
                "mean",
            ),
        )

        .reset_index()
    )

    return conv, summary


# ======================================================================
# PLOT
# ======================================================================

def plots(
    df,
    conv,
    out_dir
):

    import matplotlib.pyplot as plt

    # --------------------------------------------------------------
    # Mean summary length
    # --------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for model, g in df.groupby(
        "model"
    ):

        x = (

            g.groupby(
                "step"
            )[
                "new_tokens"
            ]

            .mean()

            .reset_index()
        )

        plt.plot(

            x["step"],

            x["new_tokens"],

            marker="o",

            label=model,
        )

    plt.xlabel(
        "Recursive step"
    )

    plt.ylabel(
        "Summary tokens"
    )

    plt.title(
        "Recursive Summary Length Drift"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(

        Path(out_dir)
        /
        "length_by_step.png",

        dpi=180,
    )

    plt.close()

    # --------------------------------------------------------------
    # growth
    # --------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    for model, g in df.groupby(
        "model"
    ):

        x = (

            g[
                g["step"] > 1
            ]

            .groupby(
                "step"
            )[
                "growth_ratio"
            ]

            .mean()

            .reset_index()
        )

        plt.plot(

            x["step"],

            x[
                "growth_ratio"
            ],

            marker="o",

            label=model,
        )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.xlabel(
        "Recursive step"
    )

    plt.ylabel(
        "S_t tokens / S_(t-1) tokens"
    )

    plt.title(
        "Recursive Growth Ratio"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(

        Path(out_dir)
        /
        "growth_by_step.png",

        dpi=180,
    )

    plt.close()


# ======================================================================
# FULL DRIFT EVAL
# ======================================================================

def eval_drift(args):

    import torch

    seed_everything(
        args.seed
    )

    chains = (
        build_drift_chains(
            args
        )
    )

    print()
    print(
        "drift chains:",
        len(chains)
    )

    save_json(

        chains,

        Path(args.out_dir)
        /
        "drift_chains.json"
    )

    rows = []

    # --------------------------------------------------------------
    # Base
    # --------------------------------------------------------------

    print()
    print("PRETRAINED")

    gen = Generator(
        args.model
    )

    rows += run_drift_model(

        "PRETRAINED",

        gen,

        chains,

        args,
    )

    del gen

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # SFT
    # --------------------------------------------------------------

    print()
    print("SFT")

    gen = Generator(

        args.model,

        str(
            Path(args.out_dir)
            /
            "qwen3_summary_lora"
        ),
    )

    rows += run_drift_model(

        "SFT",

        gen,

        chains,

        args,
    )

    del gen

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(
        rows
    )

    conv, summary = (
        aggregate_drift(
            df
        )
    )

    step = (

        df
        .groupby([
            "model",
            "step",
        ])

        .agg(

            mean_tokens=(
                "new_tokens",
                "mean",
            ),

            median_tokens=(
                "new_tokens",
                "median",
            ),

            mean_growth_ratio=(
                "growth_ratio",
                "mean",
            ),

            mean_delta_tokens=(
                "delta_tokens",
                "mean",
            ),

            mean_compression_ratio=(
                "compression_ratio",
                "mean",
            ),

            duplicate_ratio=(
                "duplicate_ratio",
                "mean",
            ),
        )

        .reset_index()
    )

    out = Path(
        args.out_dir
    )

    df.to_csv(

        out /
        "drift_detail.csv",

        index=False,
    )

    conv.to_csv(

        out /
        "drift_by_chain.csv",

        index=False,
    )

    summary.to_csv(

        out /
        "drift_summary.csv",

        index=False,
    )

    step.to_csv(

        out /
        "drift_by_step.csv",

        index=False,
    )

    plots(
        df,
        conv,
        out,
    )

    print()
    print("=" * 110)
    print("DRIFT SUMMARY")
    print("=" * 110)

    print(
        summary.to_string(

            index=False,

            float_format=
                lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 110)
    print("BY STEP")
    print("=" * 110)

    print(
        step.to_string(

            index=False,

            float_format=
                lambda x: f"{x:.4f}",
        )
    )


# ======================================================================
# ARGS
# ======================================================================

def common(p):

    p.add_argument(
        "--model",
        default=
            MODEL_NAME,
    )

    p.add_argument(
        "--out-dir",
        default=
            OUT_DIR,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=
            SEED,
    )

    p.add_argument(
        "--train-n",
        type=int,
        default=
            TRAIN_N,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=
            MAX_NEW_TOKENS,
    )

    p.add_argument(
        "--quality-eval-n",
        type=int,
        default=
            QUALITY_EVAL_N,
    )

    p.add_argument(
        "--drift-chains",
        type=int,
        default=
            DRIFT_CHAINS,
    )

    p.add_argument(
        "--drift-steps",
        type=int,
        default=
            DRIFT_STEPS,
    )


def train_args(p):

    p.add_argument(
        "--epochs",
        type=float,
        default=
            EPOCHS,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=
            BATCH_SIZE,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=
            GRAD_ACCUM,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=
            LR,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=
            MAX_LENGTH,
    )

    p.add_argument(
        "--lora-r",
        type=int,
        default=32,
    )

    p.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
    )

    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )


def args_parser():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p = sub.add_parser(
        "prepare"
    )

    common(p)

    p = sub.add_parser(
        "train"
    )

    common(p)
    train_args(p)

    p = sub.add_parser(
        "eval-quality"
    )

    common(p)

    p = sub.add_parser(
        "eval-drift"
    )

    common(p)

    p = sub.add_parser(
        "all"
    )

    common(p)
    train_args(p)

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():

    args = args_parser()

    mkdir(
        args.out_dir
    )

    if args.command == "prepare":

        prepare(args)

    elif args.command == "train":

        train(args)

    elif args.command == "eval-quality":

        eval_quality(args)

    elif args.command == "eval-drift":

        eval_drift(args)

    elif args.command == "all":

        prepare(args)

        train(args)

        eval_quality(args)

        eval_drift(args)


if __name__ == "__main__":

    main()
