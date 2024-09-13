import os
import pickle
from datetime import datetime
from sqlite3 import Connection, Cursor
from typing import Dict

import progressbar
import torch
from PIL import Image
from werkzeug.utils import secure_filename

from configs import CONSTS
from db import get_db_conn
from db_api import (
    get_exif_tag_d,
    get_sql_cols_from_d,
    get_sql_markers_from_d,
    init_db_all
)
from utils import count_image_files, get_dt_format, get_sha256

if CONSTS.clip:
    import clip

if CONSTS.ocr:
    from typing import List

    from doctr.io import DocumentFile
    from doctr.io.elements import Page
    from doctr.models import ocr_predictor
    from doctr.models.predictor.pytorch import OCRPredictor
    from numpy import ndarray


class OCRProcessor:
    def __init__(self):
        print('Loading OCR Model...')
        self.model: OCRPredictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        print('OK')

    def process(self, image_path):
        doc: List[ndarray] = DocumentFile.from_images(image_path)
        result: Page = self.model(doc)
        return result.render()


class CLIPProcessor:
    def __init__(self):
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('OK')

    def process(self, image):
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
        return image_features


class EXIFProcessor:
    @staticmethod
    def process(img: Image):
        exif = img.getexif()
        if exif:
            tags: Dict[str, str] = {get_exif_tag_d().get(k, '-'): str(v) for k, v in exif.items()}
            if '-' in tags:
                del tags['-']
            return tags
        return None


class FSProcessor:
    def __init__(self, image_path: str):
        filename_original = os.path.basename(image_path)

        self.filename_original: str = filename_original
        self.filename_secure: str = secure_filename(filename_original)
        self.filepath: str = os.path.dirname(image_path)
        self.filesize_bytes: int = os.path.getsize(image_path)
        self.sha256_digest: str = get_sha256(image_path)
        self.img: Image = Image.open(image_path)
        self.filetype: str = self.img.format.lower()


def image_id_exists_in_table(cursor: Cursor, table_name: str, image_id: int):
    sql_string = f"SELECT 1 FROM {table_name} WHERE image_id = ? LIMIT 1;"
    row = cursor.execute(sql_string, (image_id,)).fetchone()
    return row


class ImageProcessor:
    def __init__(self, ocr_processor=None, clip_processor=None, exif_processor=None):
        self.ocr_processor: OCRProcessor = ocr_processor
        self.clip_processor: CLIPProcessor = clip_processor
        self.exif_processor: EXIFProcessor = exif_processor

    def process_image(self, cursor: Cursor, image_path: str):
        fs_img = FSProcessor(image_path)

        image_id = None
        row = cursor.execute("SELECT image_id, filename_secure FROM image WHERE sha256_digest = ?;", (fs_img.sha256_digest,)).fetchone()
        if row and row.image_id:
            image_id = row.image_id

        features = {}
        if self.exif_processor and not (image_id or image_id_exists_in_table(cursor, 'exif', image_id)):
            features['exif'] = self.exif_processor.process(fs_img.img)

        if self.ocr_processor and not (image_id or image_id_exists_in_table(cursor, 'ocr', image_id)):
            features['ocr'] = self.ocr_processor.process(image_path)

        if self.clip_processor and not (image_id or image_id_exists_in_table(cursor, 'clip', image_id)):
            image = self.clip_processor.preprocess(fs_img.img).unsqueeze(0).to(CONSTS.device)
            features['clip'] = self.clip_processor.process(image)

        if features:
            store_features_in_db(cursor, image_id, fs_img, features)


def insert_feature(cursor: Cursor, image_id: int, table_name: str, feature_data: dict):
    if feature_data is not None:
        sql_args = {
            'image_id': image_id,
            **feature_data
        }
        sql = f"INSERT INTO {table_name} ({get_sql_cols_from_d(sql_args)}) VALUES ({get_sql_markers_from_d(sql_args)});"
        cursor.execute(sql, list(sql_args.values()))


def store_features_in_db(cursor: Cursor, image_id: int, fs_img: FSProcessor, features: dict):
    if not image_id:
        args = dict(
            capture_time=datetime.now().strftime(get_dt_format()),
            sha256_digest=fs_img.sha256_digest,
            filename_original=fs_img.filename_original,
            filename_secure=fs_img.filename_secure,
            filepath=fs_img.filepath,
            filesize_bytes=fs_img.filesize_bytes,
            filetype=fs_img.filetype,
        )

        sql_string = f"""
            INSERT INTO image ({get_sql_cols_from_d(args)})
            VALUES ({get_sql_markers_from_d(args)})
            ON CONFLICT(sha256_digest)
                DO UPDATE SET
                    capture_time=excluded.capture_time,
                    filename_original=excluded.filename_original,
                    filename_secure=excluded.filename_secure,
                    filepath=excluded.filepath,
                    filesize_bytes=excluded.filesize_bytes,
                    filetype=excluded.filetype
        ;"""
        cursor.execute(sql_string, list(args.values()))
        image_id = cursor.lastrowid

    if features.get('exif') is not None:
        insert_feature(cursor, image_id, 'exif', features['exif'])

    if features.get('ocr') is not None:
        insert_feature(cursor, image_id, 'ocr', {'ocr_text': features['ocr']})

    if features.get('clip') is not None:
        insert_feature(cursor, image_id, 'clip', {'features': pickle.dumps(features['clip'].cpu().numpy())})


def load_images_and_store_in_db(image_dir: str, processor: ImageProcessor):
    files_found = count_image_files(image_dir)
    files_count = min(files_found, CONSTS.limit) if CONSTS.limit else files_found

    conn: Connection = get_db_conn()
    cursor: Cursor = conn.cursor()
    with progressbar.ProgressBar(max_value=files_count) as bar:
        file_i = 0
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_i += 1
                bar.update(file_i - 1)

                if not file.lower().endswith(CONSTS.valid_extensions):
                    continue

                image_path = os.path.join(root, file)
                processor.process_image(cursor, image_path)

                if file_i > files_count:
                    break

            if file_i > files_count:
                    break

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    init_db_all()

    ocr_processor = OCRProcessor() if CONSTS.ocr else None
    clip_processor = CLIPProcessor() if CONSTS.clip else None
    exif_processor = EXIFProcessor() if CONSTS.exif else None

    processor = ImageProcessor(ocr_processor, clip_processor, exif_processor)

    load_images_and_store_in_db(CONSTS.image_dir, processor)
