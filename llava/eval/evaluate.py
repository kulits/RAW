from pathlib import Path
import argparse
import numpy as np
import orjson
import re
import scipy.optimize
import torch
import tqdm.auto as tqdm
from scipy.spatial.transform import Rotation as R

synonyms = {
    "sphere": ["sphere", "ball"],
    "cube": ["cube", "block"],
    "large": ["large", "big"],
    "small": ["small", "tiny"],
    "metal": ["metallic", "metal", "shiny"],
    "rubber": ["rubber", "matte"],
    "code": ["code", "Python code", "Python", "Python script", "script"],
    "produce": ["produce", "create", "generate", "synthesize"],
}

synonyms_inv = {n: k for k, v in synonyms.items() for n in v}

def compute_geodesic_distance_from_two_matrices(m1, m2):
    m = np.matmul(m1, m2.transpose(0, 2, 1))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = np.minimum(cos, np.ones(cos.shape))
    cos = np.maximum(cos, np.ones(cos.shape) * -1)
    return np.arccos(cos)


def or_none(x):
    return x[0] if x else None


def parse(text, n_rot=1):
    objects = []

    rot_s = "rotation="
    if n_rot != 1:
        rot_s += "\("
    rot_s += ", ".join(["([^,)]+)"] * n_rot)
    if n_rot != 1:
        rot_s += "\)"

    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("add("):
            continue

        objects.append(
            {
                "size": or_none(re.findall("size='([^',]+)'", line)),
                "color": or_none(re.findall("color='([^',]+)'", line)),
                "material": or_none(re.findall("material='([^',]+)'", line)),
                "shape": or_none(re.findall("shape='([^',]+)'", line)),
                "3d_coords": or_none(
                    re.findall("loc=\(([^,)]+), ([^,)]+), ([^,)]+)\)", line)
                ),
                "rotation": or_none(re.findall(rot_s, line)),
            }
        )

    return {
        "objects": objects,
    }


def to_num(n):
    try:
        return float(n)
    except:
        return 0


def match_euclidean(objects, objects_pred):
    cost_matrix = scipy.spatial.distance.cdist(
        np.array([n["3d_coords"] for n in objects]),
        np.array([n["3d_coords"] for n in objects_pred]),
    )
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix.T)
    return row_ind, col_ind


def main(args):
    args.out_folder.mkdir(exist_ok=True)

    if args.rotation_rep == "auto":
        if args.pred_path.stem.startswith("clevr"):
            args.rotation_rep = "yaw"
        elif "-euler-" in args.pred_path.stem:
            args.rotation_rep = "euler"
        elif "-6d-" in args.pred_path.stem:
            args.rotation_rep = "6d"
        elif "-aa-" in args.pred_path.stem:
            args.rotation_rep = "aa"
        elif "-euler_int-" in args.pred_path.stem:
            args.rotation_rep = "euler_int"
        else:
            raise ValueError(
                f"Unknown rotation representation in {args.pred_path.stem}"
            )

    n_rot = {
        "yaw": 1,
        "euler": 3,
        "euler_int": 3,
        "6d": 6,
        "aa": 3,
    }[args.rotation_rep]
    print(f'{args.rotation_rep=}, {n_rot=}')

    gt = orjson.loads(args.gt_path.read_bytes())
    if isinstance(gt, dict):
        gt = gt["scenes"]

    pred = orjson.loads(args.pred_path.read_bytes())["answers"]

    mse = []
    l2 = []
    count_acc = []
    shape_acc = []
    color_acc = []
    material_acc = []
    size_acc = []
    shape_cls_acc = []
    so3_relative_angles = []
    count_diff = []

    key_to_gt = {scene_gt["image_filename"]: scene_gt for scene_gt in gt}

    for k, v in pred.items():
        v["scene_gt"] = key_to_gt[Path(k).name]

    for k, v in tqdm.tqdm(pred.items()):
        scene_gt = v["scene_gt"]
        scene_pred = parse(v["outputs"], n_rot=n_rot)

        if len(scene_pred["objects"]) != len(scene_gt["objects"]):
            count_acc.append(0)
            count_diff.append(len(scene_pred["objects"]) - len(scene_gt["objects"]))
        else:
            count_acc.append(1)
            count_diff.append(0)

        for obj in scene_pred["objects"]:
            for k, v in obj.items():
                if k != "3d_coords" and k != "rotation":
                    obj[k] = synonyms_inv.get(v, v)

            try:
                obj["3d_coords"] = [to_num(n) for n in obj["3d_coords"]]
            except (ValueError, TypeError):
                obj["3d_coords"] = [0, 0, 0]

            try:
                obj["rotation"] = [to_num(n) for n in obj["rotation"]]
            except (ValueError, TypeError):
                obj["rotation"] = [0.0] * n_rot

            if args.rotation_rep == "6d":
                r = np.array([to_num(n) for n in obj["rotation"]]).reshape(2, 3)
                obj["rotation"] = R.from_matrix(np.vstack([r, np.cross(r[0], r[1])[None]])).as_euler('xyz').tolist()
            elif args.rotation_rep == "aa":
                obj["rotation"] = (
                    R.from_rotvec(obj["rotation"]).as_euler("xyz").tolist()
                )
            elif args.rotation_rep == "euler_int":
                obj["rotation"] = R.from_euler("XYZ", obj["rotation"]).as_euler("xyz").tolist()

        row_ind, col_ind = match_euclidean(scene_gt["objects"], scene_pred["objects"])
        objects_pred = [scene_pred["objects"][i] for i in row_ind]
        objects_gt = [scene_gt["objects"][i] for i in col_ind]

        try:
            shape_acc.append(
                np.mean(
                    [a["shape"] == b["shape"] for a, b in zip(objects_gt, objects_pred) if a['shape'].lower() != 'sphere']
                )
            )
        except KeyError:
            shape_acc.append(np.nan)

        try:
            color_acc.append(
                np.mean(
                    [a["color"] == b["color"] for a, b in zip(objects_gt, objects_pred)]
                )
            )
        except KeyError:
            color_acc.append(np.nan)

        try:
            material_acc.append(
                np.mean(
                    [
                        a["material"] == b["material"]
                        for a, b in zip(objects_gt, objects_pred)
                    ]
                )
            )
        except KeyError:
            material_acc.append(np.nan)

        try:
            size_acc.append(
                np.mean(
                    [a["size"] == b["size"] for a, b in zip(objects_gt, objects_pred)]
                )
            )
        except KeyError:
            size_acc.append(np.nan)

        try:
            shape_cls_acc.append(
                np.mean(
                    [
                        a["shape"].split("_")[0] == str(b["shape"]).split("_")[0]
                        for a, b in zip(objects_gt, objects_pred)
                    ]
                )
            )
        except KeyError:
            shape_cls_acc.append(np.nan)

        mse.append(
            torch.nn.functional.mse_loss(
                torch.tensor([n["3d_coords"] for n in objects_gt]),
                torch.tensor([n["3d_coords"] for n in objects_pred]),
            )
        )
        l2.append(
            torch.nn.functional.pairwise_distance(
                torch.tensor([n["3d_coords"] for n in objects_gt]),
                torch.tensor([n["3d_coords"] for n in objects_pred]),
                p=2,
            ).mean()
        )

        if n_rot > 1:
            so3_relative_angles.append(
                (
                    compute_geodesic_distance_from_two_matrices(
                        np.array(
                            [
                                R.from_euler("xyz", n["rotation"]).as_matrix()
                                for n in objects_gt
                            ]
                        ),
                        np.array(
                            [
                                R.from_euler("xyz", n["rotation"]).as_matrix()
                                for n in objects_pred
                            ]
                        ),
                    )
                    .mean()
                    .item()
                    / np.pi
                    * 180
                )
            )

    print(
        *(f'{k}={v:.2f}' for k, v in {
            "l2": torch.tensor([n for n in l2 if n]).mean().item(),
            "geod": torch.tensor(so3_relative_angles).mean().item(),
            "ace": np.abs(count_diff).mean(),
            "size": 100*torch.tensor(size_acc).to(torch.float).mean().item(),
            "color": 100*torch.tensor(color_acc).to(torch.float).mean().item(),
            "mat.": 100*torch.tensor(material_acc).to(torch.float).mean().item(),
            "shape": 100*torch.tensor(shape_acc).to(torch.float).nanmean().item(),
            "shp_cls": 100*torch.tensor(shape_cls_acc).to(torch.float).mean().item(),
        }.items())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--out_folder", type=Path, default=Path("./eval"))
    parser.add_argument("--rotation_rep", type=str, default="auto")
    args = parser.parse_args()

    assert args.gt_path.exists(), f"File {args.gt_path} does not exist."
    assert args.pred_path.exists(), f"File {args.pred_path} does not exist."

    main(args)
