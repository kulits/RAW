from pathlib import Path
from subprocess import run
import argparse
import socket

print(__file__)

with socket.socket() as s:
    s.bind(("", 0))
    main_process_port = s.getsockname()[1]

prompt_version = "v1"
model_version = "336px-pretrain-vicuna-7b-v1.3"

parser = argparse.ArgumentParser()
parser.add_argument("--images_tar", type=str, required=True)
parser.add_argument("--images_val_tar", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--data_path_val", type=str)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--emb_path", type=str, required=True)
parser.add_argument("--deepspeed", default="./scripts/zero2.json", type=str)
parser.add_argument("--lora_enable", default=True, type=bool)
parser.add_argument("--lora_r", default=128, type=int)
parser.add_argument("--lora_alpha", default=256, type=int)
parser.add_argument("--model_name_or_path", default="lmsys/vicuna-7b-v1.3", type=str)
parser.add_argument("--version", default=prompt_version, type=str)
parser.add_argument("--image_folder", default="/tmp/images", type=str)
parser.add_argument("--image_folder_val", default="/tmp/images_val", type=str)
parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
parser.add_argument(
    "--pretrain_mm_mlp_adapter", default=f"./checkpoints/llava-{model_version}/mm_projector.bin", type=str
)
parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
parser.add_argument("--mm_use_im_start_end", default=False, type=bool)
parser.add_argument("--mm_use_im_patch_token", default=False, type=bool)
parser.add_argument("--bf16", default=True, type=bool)
parser.add_argument("--bits", default=16, type=int)
parser.add_argument("--num_train_epochs", default=30, type=int)
parser.add_argument("--max_steps", type=int)
parser.add_argument("--per_device_train_batch_size", default=32, type=int)
parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--save_strategy", default="steps", type=str)
parser.add_argument("--save_steps", default=500, type=int)
parser.add_argument("--save_total_limit", default=1, type=int)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--mm_projector_lr", default=2e-5, type=float)
parser.add_argument("--float_head_lr", default=2e-4, type=float)
parser.add_argument("--warmup_ratio", default=0.03, type=float)
parser.add_argument("--lr_scheduler_type", default="cosine", type=str)
parser.add_argument("--logging_steps", default=1, type=int)
parser.add_argument("--tf32", default=True, type=bool)
parser.add_argument("--model_max_length", default=2048, type=int)
parser.add_argument("--gradient_checkpointing", default=True, type=bool)
parser.add_argument("--lazy_preprocess", default=True, type=bool)
parser.add_argument("--dataloader_num_workers", default=20, type=int)
parser.add_argument("--report_to", default="wandb", type=str)
parser.add_argument("--evaluation_strategy", default="steps", type=str)
parser.add_argument("--eval_steps", default=100, type=int)
parser.add_argument("--mm_projector_type", type=str)
parser.add_argument("--image_aspect_ratio", default="square", type=str)
parser.add_argument("--num_samples", default=-1, type=int)
parser.add_argument("--appearance_dim", default=1024, type=int)
parser.add_argument("--use_height", default=1, type=int)
parser.add_argument("--use_prompt", default=1, type=int)
parser.add_argument("--eval_num_objects", default=5, type=int)
parser.add_argument("--text_retrieve", default=0, type=int)
parser.add_argument("--fuzz_env_params", default=0, type=int)
parser.add_argument("--no_env", default=0, type=int)
parser.add_argument("--use_qualified_add", default=1, type=int)
parser.add_argument("--use_pixel_count", default=1, type=int)

args = parser.parse_args()

assert Path(args.images_tar).exists()
assert Path(args.images_val_tar).exists()

Path("/tmp/images").mkdir(exist_ok=True)
Path("/tmp/images-val").mkdir(exist_ok=True)

run(["tar", "xf", args.images_tar, "-C", args.image_folder], check=True)
run(["tar", "xf", args.images_val_tar, "-C", args.image_folder_val], check=True)

if 'convnext' in args.emb_path:
    emb_prefix = 'convnext'
elif 'dinov2-giant' in args.emb_path:
    emb_prefix = 'dinov2-giant'
elif 'bioclip' in args.emb_path:
    emb_prefix = 'bioclip'
else:
    raise ValueError("Unknown emb type.")

run(["rsync", "-a", args.data_path, "/tmp/train.feather"], check=True)
args.data_path = "/tmp/train.feather"

run(["rsync", "-a", args.data_path_val, "/tmp/val.feather"], check=True)
args.data_path_val = "/tmp/val.feather"

run(["rsync", "-a", args.emb_path, "/tmp/emb.npz"], check=True)
args.emb_path = "/tmp/emb.npz"

run(["rsync", "-a", f"./data/floor_{emb_prefix}.npz", "/tmp/floor_emb.npz"], check=True)
args.floor_emb_path = "/tmp/floor_emb.npz"

run(["rsync", "-a", f"./data/env_params.npz", "/tmp/env_params.npz"], check=True)
args.env_params_path = "/tmp/env_params.npz"

del args.images_tar
del args.images_val_tar

run(
    [
        "deepspeed",
        "--master_port",
        str(main_process_port),
        "llava/train/train.py",
    ]
    + [f"--{k}={v}" for k, v in vars(args).items() if v is not None],
    env={
        "PYTHONPATH": ".",
    },
    check=True,
)
