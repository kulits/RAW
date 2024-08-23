<h1 align=center>Re-Thinking Inverse Graphics With Large Language Models</h1>

<p align=center><a href="https://kulits.github.io/">Peter Kulits</a><sup>*</sup>, <a href="https://ps.is.mpg.de/person/hfeng">Haiwen Feng</a><sup>*</sup>, <a href="https://wyliu.com/">Weiyang Liu</a>, <a href="https://is.mpg.de/~vabrevaya">Victoria Abrevaya</a>, <a href="https://ps.is.mpg.de/person/black">Michael J. Black</a></p>

<p align=center><a href="https://ig-llm.is.tue.mpg.de">[Project Page]</a> <a href="https://openreview.net/forum?id=u0eiu1MTS7">[TMLR]</a></p>

<h2>Summary</h2>
<em>We present the Inverse-Graphics Large Language Model (<b>IG-LLM</b>) framework, a general approach to solving inverse-graphics problems. We instruction-tune an LLM to decode a visual (CLIP) embedding into graphics code that can be used to reproduce the observed scene using a standard graphics engine. Leveraging the broad reasoning abilities of LLMs, we demonstrate that our framework exhibits natural generalization across a variety of distribution shifts without the use of special inductive biases.</em>
<br/><br/>

![image](https://ig-llm.is.tue.mpg.de/media/upload/header.jpeg)

<h2>Data</h2>
Training and evaluation data can be found at <a href="https://ig-llm.is.tue.mpg.de/download.php">https://ig-llm.is.tue.mpg.de/download.php</a> after registering on the <a href="https://ig-llm.is.tue.mpg.de/">project page</a>. The following is an outline of the data available:
<details>

```sh
├── CLEVR
│   ├── images
│   │   ├── train.tar
│   │   ├── val_ID.tar
│   │   └── val_OOD.tar
│   └── labels
│       ├── train.json
│       ├── val_ID.json
│       └── val_OOD.json
├── 2D
│   └── 2d.npz
├── SO3
│   ├── images
│   │   ├── train.tar
│   │   ├── val_ID.tar
│   │   └── val_OOD.tar
│   └── labels
│       ├── train.json
│       ├── val_ID.json
│       └── val_OOD.json
├── 6DoF
│   ├── images
│   │   ├── train.tar
│   │   └── val_ID.tar
│   └── labels
│       ├── train.json
│       └── val_ID.json
└── ShapeNet
    ├── images
    │   ├── train.tar
    │   ├── val_ID.tar
    │   ├── val_OOD_texture.tar
    │   └── val_OOD_shape.tar
    └── labels
        ├── train.json
        ├── val_ID.json
        ├── val_OOD_texture.json
        └── val_OOD_shape.json
```
</details>

<h2>Setup</h2>
The environment can be configured with <code>conda</code>/<code>micromamba</code> from <code>environment.yml</code> or using the <code>Dockerfile</code>.

<h2>Training</h2>
After the data has been downloaded, training can be initiated with the following:

<ul>
<li><b>CLEVR</b>
<details>

```sh
python train.py \
    --images_tar data/CLEVR/images/train.tar \
    --data_path data/CLEVR/images/train.json \
    --images_val_tar data/CLEVR/images/val_OOD.tar \
    --data_path_val data/CLEVR/labels/val_OOD.json \
    --per_device_train_batch_size X \
    --output_dir ./checkpoints/clevr-Y \
    --max_steps 40000 \
    --float_head_type (none|tanh_mlp_gelu) \
    --image_aspect_ratio pad \
    --num_samples 4000
```
</details>
</li>

<li><b>2D</b>
<details>
<code>2d.npz</code> is expected to be at <code>data/2d.npz</code> prior to running <code>train.py</code>.

```sh
python train.py \
    --data_path checkerboard_sparse \
    --data_path_val random \
    --per_device_train_batch_size X \
    --output_dir ./checkpoints/2d-Y \
    --max_steps 40000 \
    --float_head_type (none|tanh_mlp_gelu) \
    --image_aspect_ratio pad \
    --is_2d True
```
</details>
</li>

<li><b>SO(3)</b>
<details>

```sh
python train.py \
    --images_tar data/SO3/images/train.tar \
    --data_path data/SO3/images/train.json \
    --images_val_tar data/SO3/images/val_OOD.tar \
    --data_path_val data/SO3/labels/val_OOD.json \
    --per_device_train_batch_size X \
    --output_dir ./checkpoints/so3-Y \
    --max_steps 40000 \
    --float_head_type (none|tanh_mlp_gelu) \
    --image_aspect_ratio pad \
    --rotation_rep (euler_int|euler|aa|6d)
```
</details>
</li>

<li><b>6-DoF</b>
<details>

```sh
python train.py \
    --images_tar data/6DoF/images/train.tar \
    --data_path data/6DoF/images/train.json \
    --images_val_tar data/6DoF/images/val_ID.tar \
    --data_path_val data/6DoF/labels/val_ID.json \
    --per_device_train_batch_size X \
    --output_dir ./checkpoints/6dof-Y \
    --max_steps 200000 \
    --float_head_type (none|tanh_mlp_gelu) \
    --image_aspect_ratio pad \
    --rotation_rep (euler_int|euler|aa|6d)
```
</details>
</li>

<li><b>ShapeNet</b>
<details>

```sh
python train.py \
    --images_tar data/ShapeNet/images/train.tar \
    --data_path data/ShapeNet/images/train.json \
    --images_val_tar data/ShapeNet/images/val_OOD_texture.tar \
    --data_path_val data/ShapeNet/labels/val_OOD_texture.json \
    --per_device_train_batch_size X \
    --output_dir ./checkpoints/shapenet-Y \
    --max_steps 500000 \
    --float_head_type (none|tanh_mlp_gelu) \
    --image_aspect_ratio pad \
    --rotation_rep (euler_int|euler|aa|6d)
```
</details>
</li>
</ul>

<h3>Inference</h3>

```sh
python inference.py \
    --model-path ./checkpoints/clevr-Y \
    --images_tar data/CLEVR/images/val_OOD.tar \
    --out_path ./out/clevr-Y-val_OOD.json \
    --image_aspect_ratio pad
```

<h3>License</h3>
We build off the <a href="https://github.com/haotian-liu/LLaVA">LLaVA</a> codebase to perform our experiments. As such, inherited code falls under the original Apache 2.0 license. Additions and modifications are released under a different license in accordance with institute requirements which has been prepended to <code>LICENSE</code>.
