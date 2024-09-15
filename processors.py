import os
import pickle
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection, Cursor
from typing import Dict, Generator

import tqdm
import torch
from PIL import Image
from werkzeug.utils import secure_filename

from configs import CONSTS
from consts import (
    clip_valid_extensions,
    exif_valid_extensions,
    ocr_valid_extensions,
    valid_extensions
)
from db import get_db_conn
from db_api import (
    get_exif_tag_d,
    get_sql_cols_from_d,
    get_sql_markers_from_d,
    init_db_all, 
    query_db
)
from ocr import OCRBase, OCRDoctr, OCRRobertKnight, OCRTerreract
from utils import count_image_files, get_dt_format, get_sha256

if CONSTS.clip:
    import clip
from time import perf_counter

class OCRProcessor:
    def __init__(self, ocr_type) -> None:
        self.ocr_type = ocr_type
        self.obj: OCRBase = {'ocrs': OCRRobertKnight, 'tesseract': OCRTerreract, 'doctr': OCRDoctr}[self.ocr_type]()
    
    def process(self, image_path):
        return self.obj.process(image_path)


class CLIPProcessor:
    def __init__(self):
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('Finished')

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

        self.sha256_digest_to_image_id: dict = {row.sha256_digest: row.image_id for row in query_db("""SELECT image_id, sha256_digest FROM image;""")}
        
        print('Setting up sets.')
        self.clip_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM clip;""")} if CONSTS.clip else None
        self.exif_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM exif;""")} if CONSTS.exif else None
        self.ocr_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM ocr;""")} if CONSTS.ocr else None
        print('Finished.')

    def process_image(self, cursor: Cursor, image_path: str):
        fs_img = FSProcessor(image_path)

        image_id = self.sha256_digest_to_image_id.get(fs_img.sha256_digest, None)
        features = {}

        if self.exif_processor and (image_id not in self.exif_image_ids):
            features['exif'] = self.exif_processor.process(fs_img.img)

        if self.ocr_processor and (image_id not in self.ocr_image_ids):
            features['ocr'] = self.ocr_processor.process(image_path)

        if self.clip_processor and (image_id not in self.clip_image_ids):
            image = self.clip_processor.preprocess(fs_img.img).unsqueeze(0).to(CONSTS.device) # 0.08s
            features['clip'] = self.clip_processor.process(image) # 0.03s

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


def get_image_paths(root_dir) -> Generator:
    return (str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in valid_extensions)


def load_images_and_store_in_db(root_image_folder: str, processor: ImageProcessor):
    file_paths = get_image_paths(root_image_folder)
    files_found = count_image_files(root_image_folder)
    files_count = min(files_found, CONSTS.processor_file_limit) if CONSTS.processor_file_limit else files_found

    conn: Connection = get_db_conn()
    cursor: Cursor = conn.cursor()
    for i, file_path in enumerate(tqdm.tqdm(file_paths, total=files_count)):

        try:
            processor.process_image(cursor, file_path)
        except Exception as e:
            print(e)

        if i >= files_count:
            break

        if CONSTS.ocr and i % 320 == 0:
            conn.commit() # ocr is computationally expensive.
        elif not CONSTS.ocr and (CONSTS.clip or CONSTS.exif) and i % 2000 == 0:
            conn.commit()

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    init_db_all()

    ocr_processor = OCRProcessor(CONSTS.ocr_type) if CONSTS.ocr else None
    clip_processor = CLIPProcessor() if CONSTS.clip else None
    exif_processor = EXIFProcessor() if CONSTS.exif else None

    processor = ImageProcessor(ocr_processor, clip_processor, exif_processor)

    load_images_and_store_in_db(CONSTS.root_image_folder, processor)
