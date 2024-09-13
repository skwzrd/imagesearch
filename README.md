# EclipseImg

Get your venv up with and install the `requirements.txt`. Consider following the guides on the dependency repos.

Set your configs in `configs.py`.

Run `python3.12 processors.py` to crawl images and populate a sqlite database with CLIP embeddings, OCR text, and/or EXIF data.

Run `python3.12 web.py` to launch the web UI for searching the sqlite database.

## Dependencies

- OCR [https://github.com/mindee/doctr](https://github.com/mindee/doctr)
- Search Embeddings [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

## Preview

![preview](preview.png)