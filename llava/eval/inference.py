import argparse
import gzip
import os
from pathlib import Path

import numpy as np
import orjson
import torch
import tqdm.auto as tqdm
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image, UnidentifiedImageError


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, device_map=args.device
    )

    answers = {}
    for image_path in tqdm.tqdm(
        sorted(Path(args.image_folder).rglob("*"))
    ):
        try:
            image = Image.open(image_path)
        except (IsADirectoryError, UnidentifiedImageError):
            continue

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(
            conv.roles[0],
            "<image>",
        )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image = image.convert("RGB")
        if args.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
        else:
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

        with torch.inference_mode():
            output_ids, rotations, appearances = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=args.num_beams,
                max_new_tokens=2048,
                use_cache=True,
                bad_words_ids=tokenizer(
                    [
                        "(\n",
                        "( ",
                        " )",
                    ],
                    add_special_tokens=False,
                ).input_ids,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(image_path, outputs, flush=True)

        answers[f"{image_path}"] = {
            "outputs": outputs,
            "rotations": rotations.tolist() if rotations is not None else None,
            "appearances": appearances.tolist() if appearances is not None else None,
        }

    Path(args.out_path).write_bytes(
        gzip.compress(
            orjson.dumps(
                {
                    "model_path": args.model_path,
                    "model_base": args.model_base,
                    "model_name": args.model_name,
                    "answers": answers,
                },
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        print(f"File {args.out_path} already exists. Skipping.")
        exit(0)

    main(args)
