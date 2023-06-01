import argparse
import json
import os

import transformers

from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative
from PrefixGPT2 import PrefixGPT2

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="GPT2LMHeadModel",
    choices=[
        "BertForMaskedLM",
        "AlbertForMaskedLM",
        "RobertaForMaskedLM",
        "GPT2LMHeadModel",
    ],
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="gpt2",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--save_path",
    action="store",
    type=str
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)
parser.add_argument(
    "--split",
    action="store",
    type=str,
    default='test'
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="stereoset",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        seed=args.seed,
    )

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")

    model = PrefixGPT2.from_pretrained(args.save_path)
    # model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/{args.split}.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model),
    )
    results = runner()

    os.makedirs(f"{args.save_path}/results/stereoset", exist_ok=True)
    predictions_file = f"{args.save_path}/results/stereoset/{args.split}.json"
    with open(
        predictions_file, "w"
    ) as f:
        json.dump(results, f, indent=2)
    output_file = f"{args.save_path}/results/stereoset/stereoset_final_{args.split}_results.json"
    command = f'python experiments/stereoset_evaluation.py --predictions_file {predictions_file} --output_file {output_file} --split {args.split}'
    os.system(command)