#!/usr/bin/python3

import argparse
import shutil
import sys
from pathlib import Path
from subprocess import run
import json
import gzip


prompt_version = "v1"
model_version = "336px-pretrain-vicuna-7b-v1.3"

parser = argparse.ArgumentParser()
parser.add_argument("--images_tar", type=str)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.3")
parser.add_argument("--model-name", type=str, default="llava-lora")
parser.add_argument("--image-folder", type=str, default="/tmp/images-val")
parser.add_argument("--out_path", type=str)
parser.add_argument("--conv-mode", type=str, default="llava_v1")
parser.add_argument("--slice-start", type=int, default=None)
parser.add_argument("--slice-end", type=int, default=None)
parser.add_argument("--slice-step", type=int, default=None)
parser.add_argument("--image_aspect_ratio", type=str, default="square")
parser.add_argument("--is_2d", default=0, type=int)
parser.add_argument("--annotations_path", type=str, default=None)

args = parser.parse_args()


assert args.images_tar is not None, args.images_tar
assert Path(args.images_tar).exists(), args.images_tar

if args.annotations_path is not None:
    assert Path(args.annotations_path).exists(), args.annotations_path

Path("/tmp/images-val").mkdir()

if args.annotations_path is not None:
    with gzip.open(args.annotations_path) as f:
        image_ids = {a["image_id"] for a in json.load(f)[slice(args.slice_start, args.slice_end, args.slice_step)]}

    print(len(image_ids), flush=True)

print("Extracting images", flush=True)
run(["tar", "xf", args.images_tar, "-C", "/tmp/images-val/"], check=True)

if args.annotations_path is not None:
    for image_path in Path("/tmp/images-val").glob("*"):
        if image_path.name not in image_ids:
            image_path.unlink()
elif args.slice_start is not None or args.slice_end is not None or args.slice_step is not None:
    valid_paths = sorted(Path("/tmp/images-val").glob("*"))[slice(args.slice_start, args.slice_end, args.slice_step)]
    for image_path in Path("/tmp/images-val").glob("*"):
        if image_path not in valid_paths:
            image_path.unlink()
print("Extracted images", flush=True)

del args.images_tar
del args.slice_start
del args.slice_end
del args.slice_step
del args.annotations_path

run(
    [
        "python",
        "llava/eval/inference.py",
    ]
    + [f"--{k}={v}" for k, v in vars(args).items() if v is not None],
    env={
        "PYTHONPATH": ".",
    },
    check=True,
)
