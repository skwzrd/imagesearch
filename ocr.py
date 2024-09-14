import re

from configs import CONSTS

if CONSTS.ocr:

    if CONSTS.ocr_type == 'doctr':
        from typing import List

        from doctr.io import DocumentFile
        from doctr.io.elements import Page
        from doctr.models import ocr_predictor
        from doctr.models.predictor.pytorch import OCRPredictor
        from numpy import ndarray

    elif CONSTS.ocr_type == 'tesseract':
        import string

        import pytesseract
    
    elif CONSTS.ocr_type == 'ocrs':
        import subprocess

    else:
        raise ValueError('CONSTS.ocr_type not specified in configs.py')


def apply_text_filter(func):
    def wrapper(self, *args, **kwargs):
        text = func(self, *args, **kwargs)
        return self.text_filter(text)
    return wrapper


class OCRBase:
    def __init__(self):
        pass

    def process(self):
        pass

    def text_filter(self, text):
        return re.sub(r'(?m)^\s*\S{1,3}\s*$\n?', '', text).strip()


class OCRDoctr(OCRBase):
    """
    python3.12 -m pip install doctr
    """
    def __init__(self):
        print('Loading OCR Model...')
        self.model: OCRPredictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        print('Finished')

    @apply_text_filter
    def process(self, image_path):
        doc: List[ndarray] = DocumentFile.from_images(image_path)
        result: Page = self.model(doc)
        return result.render()


class OCRTerreract(OCRBase):
    """
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install libtesseract-dev
    
    python3.12 -m pip install pytesseract
    """
    def __init__(self):
        whitelist = string.printable.replace('"', '\\"').replace("'", "\\'")
        self.config = fr'-c tessedit_char_whitelist=" {whitelist}"'
        pass

    @apply_text_filter
    def process(self, image_path):
        text = pytesseract.image_to_string(image_path, config=self.config, timeout=20)
        return text


class OCRRobertKnight(OCRBase):
    """
    curl https://sh.rustup.rs -sSf | sh
    sudo apt install cargo
    cargo install ocrs-cli
    """
    def __init__(self):
        pass

    @apply_text_filter
    def process(self, image_path):
        text = subprocess.run(['ocrs', image_path], capture_output=True, text=True).stdout
        return text
