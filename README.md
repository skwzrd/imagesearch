# EclipseImg

Get your venv up with and install the `requirements.txt`.
- Run `python3.12 -m pip install git+https://github.com/openai/CLIP.git` to install CLIP.
- Run `python3.12 -m pip install -r requirements.txt`
- If you don't have nvidia drivers installed, you can use `sudo ubuntu-drivers autoinstall` and reboot.

Set your configs in `configs.py`.

Run `python3.12 processors.py` to crawl images and populate a sqlite database with CLIP embeddings, OCR text, and/or EXIF data.

Run `python3.12 web.py` to launch the web UI for searching the sqlite database.

## Dependencies

- For searching embeddings, we use [https://github.com/openai/CLIP](https://github.com/openai/CLIP) (see `requirements.txt` for install).

If you want OCR, pick one OCR program.

- ocrs (recommended) (download and compile https://github.com/robertknight/ocrs)
- doctr (see `requirements.txt`)
- pytesseract (see `requirements.txt`)

Need help choosing? See [https://neetventures.com/post/50](https://neetventures.com/post/50).

## Preview

![preview](preview.png)