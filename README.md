# ImageSearch

A program that stiches together many libraries in order to provide a new image search experience.

![preview](preview.png)

Get your venv up with and install the `requirements.txt`.
- Run `python3.12 -m pip install git+https://github.com/openai/CLIP.git` to install CLIP.
- Run `python3.12 -m pip install -r requirements.txt`
- If you don't have nvidia drivers installed, you can use `sudo ubuntu-drivers autoinstall` and reboot.

Set your configs in `configs.py`.

Run `python3.12 process.py` to crawl images and populate a sqlite database with CLIP embeddings, OCR text, EXIF data, Hashes, Noise, and/or Faces.

Run `python3.12 web.py` to launch the web UI for searching the sqlite database.


## Dependencies

- For searching embeddings, we use [https://github.com/openai/CLIP](https://github.com/openai/CLIP) (see `requirements.txt` for install).

If you want OCR, pick one OCR program.

- ocrs (recommended) (download and compile [https://github.com/robertknight/ocrs](https://github.com/robertknight/ocrs))
- doctr (see `requirements.txt`)
- pytesseract (see `requirements.txt`)

If you want facial detection to search by face counts, install `face_recognition` (see `requirements.txt` ).

Need help choosing? See [https://neetventures.com/post/50](https://neetventures.com/post/50).


## Performance

My server (see below) does 160 images/second with EXIF, CLIP, and OCR ([https://github.com/robertknight/ocrs](https://github.com/robertknight/ocrs)) engaged.


## Hardware Requirements

These metrics were taken while processing images. I'll let you interpret them.

`sudo lshw -class CPU | grep -i product`

```
product: Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz
```

`free -mh`

```
               total        used        free      shared  buff/cache   available
Mem:            31Gi       2.5Gi        20Gi        17Mi       8.0Gi        28Gi
```

`nvidia-smi`

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.107.02             Driver Version: 550.107.02     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1660 Ti     Off |   00000000:01:00.0 Off |                  N/A |
| 23%   41C    P2             28W /  120W |     466MiB /   6144MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     25919      C   python3.12                                    462MiB |
+-----------------------------------------------------------------------------------------+
```
