<h1 align="center">Reconstructing Animals <em>and</em> the Wild </h1>
<p align="center">
  <a href="https://kulits.github.io/">Peter Kulits</a>, <a href="https://ps.is.mpg.de/person/black">Michael J. Black</a>, <a href="https://imati.cnr.it/mypage.php?idk=PG-2">Silvia Zuffi</a>
</p>
<p align="center">
  <a href="https://raw.is.tue.mpg.de">[Project Page]</a>
</p>
<p>Data and code coming soon.</p>
<h2>Summary</h2>We train an LLM to decode a frozen CLIP embedding of a natural image into a structured compositional scene representation encompassing both animals <em>and</em> their habitats.
<h3 style="text-align:center">
  <span>
    <img src="https://raw.is.tue.mpg.de/media/upload/0.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/1.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/2.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/3.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/4.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/5.jpg" width="130" height="130" alt="">
  </span>
  <span>
    <img src="https://raw.is.tue.mpg.de/media/upload/0P.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/1P.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/2P.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/3P.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/4P.jpg" width="130" height="130" alt="">
    <img src="https://raw.is.tue.mpg.de/media/upload/5P.jpg" width="130" height="130" alt="">
  </span>
</h3>

<h2>Data</h2>
Data can be found at <a href="https://raw.is.tue.mpg.de/download.php">https://raw.is.tue.mpg.de/download.php</a> after registering on the <a href="https://raw.is.tue.mpg.de/">project page</a>.

<h2>Setup</h2>
The environment can be configured with <code>conda</code>/<code>micromamba</code> from <code>environment.yml</code> or using the <code>Dockerfile</code>.

<h2>Training</h2>
After the data has been downloaded, training can be initiated with the following:

```sh
python train.py \
    --images_tar data/train.tar \
    --data_path data/train.gz.feather \
    --images_val_tar data/val.tar \
    --data_path_val data/val.gz.feather \
    --per_device_train_batch_size X \
    --output_dir ./checkpoints/RAW-Y \
    --max_steps 40000 \
    --image_aspect_ratio pad
```

<h3>Inference</h3>

```sh
python inference.py \
    --model-path ./checkpoints/RAW-Y \
    --images_tar data/val.tar \
    --out_path ./out/RAW-Y.json.gz \
    --image_aspect_ratio pad
```

<h3>License</h3>
We build off the <a href="https://github.com/haotian-liu/LLaVA">LLaVA</a> codebase to perform our experiments. As such, inherited code falls under the original Apache 2.0 license. Additions and modifications are released under a different license in accordance with institute requirements which has been prepended to <code>LICENSE</code>.
