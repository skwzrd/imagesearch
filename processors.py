import os
import pickle
import sqlite3
from collections import OrderedDict
from datetime import datetime
from itertools import batched
from pathlib import Path
from sqlite3 import Cursor
from typing import Dict, Generator, List

import torch
import tqdm
from PIL import Image
from werkzeug.utils import secure_filename

from consts import CONSTS, processor_types, valid_extensions
from db_api import (
    get_exif_tag_d,
    get_sql_cols_from_d,
    get_sql_markers_from_d,
    init_db_all,
    query_db
)
from utils import Perf, count_image_files, get_dt_format, get_sha256

if CONSTS.ocr:
    if CONSTS.ocr_type == 'ocrs':
        from ocr import OCRBase
        from ocr import OCRRobertKnight as OCR
    if CONSTS.ocr_type == 'doctr':
        from ocr import OCRBase
        from ocr import OCRDoctr as OCR
    if CONSTS.ocr_type == 'tesseract':
        from ocr import OCRBase
        from ocr import OCRTerreract as OCR

if CONSTS.clip:
    import clip

if CONSTS.hash:
    import imagehash

if CONSTS.face:
    import face_recognition
    from numpy import array


class OCRProcessor:
    def __init__(self, ocr_type) -> None:
        self.ocr_type = ocr_type
        self.obj: OCRBase = OCR()
    
    def process(self, image_path) -> dict:
        return {'ocr_text': self.obj.process(image_path)}


class HashProcessor:
    def __init__(self, hash_dict: OrderedDict|None=None) -> None:
        # keys must be equal to a hash table column name
        # you can comment out the hashes you don't want
        self.default_hash_dict = OrderedDict()

        if CONSTS.hash_average: self.default_hash_dict['average_hash'] = imagehash.average_hash
        if CONSTS.hash_color: self.default_hash_dict['colorhash'] = imagehash.colorhash
        if CONSTS.hash_crop_resistant: self.default_hash_dict['crop_resistant_hash'] = imagehash.crop_resistant_hash

        if hash_dict is not None and len(hash_dict) > 0:
            for k in hash_dict:
                if k not in self.default_hash_dict:
                    raise ValueError(k)
            self.hash_dict = hash_dict
        else:
            self.hash_dict = self.default_hash_dict

    def process(self, img: Image) -> dict:
        if not img:
            raise ValueError(f'{img=}')
        return {k: pickle.dumps(hashfunc(img)) for k, hashfunc in self.hash_dict.items()} # BLOBs for each hash func


class FSProcessor:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.sha256_digest: str = get_sha256(self.image_path)

        self.filename_original: str = None
        self.filename_secure: str = None
        self.filepath: str = None
        self.filesize_bytes: int = None
        self.sha256_digest: str = None
        self.img: Image = None
        self.filetype: str = None

        self.processed = False

    def process(self):
        """We may not need to process all of this when our object is first instantiated."""
        if not self.processed:
            self.filename_original: str = os.path.basename(self.image_path)
            self.filename_secure: str = secure_filename(self.filename_original)
            self.filepath: str = os.path.dirname(self.image_path)
            self.filesize_bytes: int = os.path.getsize(self.image_path)
            self.img: Image = Image.open(self.image_path)
            self.filetype: str = self.img.format.lower()
            
            self.processed = True


def save_images(img_array, face_locations, filename_secure):
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = img_array[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(os.path.join(os.path.dirname(__file__), 'ignore', f'{filename_secure}___{str(i).zfill(3)}.png'))


class FaceProcessor:
    def __init__(self) -> None:
        pass

    def process(self, img: Image, filename_secure: str) -> dict:
        img_array = array(img.convert('RGB'))
        model = 'hog' if CONSTS.device == 'cuda' else 'hog'
        face_locations = face_recognition.face_locations(img_array, model=model) # 1-2 images/s

        face_count = len(face_locations)
        face_encodings = None

        if face_count:
            if CONSTS.face_save:
                save_images(img_array, face_locations, filename_secure)

            if CONSTS.face_encodings:
                face_encodings: List[array] = face_recognition.face_encodings(img_array, face_locations)

        face_encodings = pickle.dumps(face_encodings)

        return dict(face_count=face_count, face_encodings=face_encodings)


class CLIPProcessor:
    def __init__(self) -> None:
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('Finished')

    def process(self, images: list) -> list[dict]:
        processed_images = torch.stack([self.preprocess(img) for img in images]).to(CONSTS.device)

        features_list = []
        with torch.no_grad():
            features_list.extend(self.model.encode_image(processed_images).float().cpu().numpy())

        return [{'features': pickle.dumps(features)} for features in features_list]


class EXIFProcessor:
    @staticmethod
    def process(img: Image) -> dict:
        exif = img.getexif()
        if exif:
            tags: Dict[str, str] = {get_exif_tag_d().get(k, '-'): str(v) for k, v in exif.items()}
            if '-' in tags:
                del tags['-']
            return tags
        return None


def image_id_exists_in_table(cursor: Cursor, table_name: str, image_id: int):
    sql_string = f"SELECT 1 FROM {table_name} WHERE image_id = ? LIMIT 1;"
    row = cursor.execute(sql_string, (image_id,)).fetchone()
    return row


class ImageProcessor:
    def __init__(self, ocr_processor=None, clip_processor=None, exif_processor=None, hash_processor=None, face_processor=None):
        self.ocr_processor: OCRProcessor = ocr_processor
        self.clip_processor: CLIPProcessor = clip_processor
        self.exif_processor: EXIFProcessor = exif_processor
        self.hash_processor: HashProcessor = hash_processor
        self.face_processor: FaceProcessor = face_processor

        print('Setting up sets...')
        self.sha256_digest_to_image_id: dict = {row.sha256_digest: row.image_id for row in query_db("""SELECT image_id, sha256_digest FROM image;""")}

        self.clip_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM clip;""")} if CONSTS.clip else None
        self.exif_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM exif;""")} if CONSTS.exif else None
        self.ocr_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM ocr;""")} if CONSTS.ocr else None
        self.hash_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM hash;""")} if CONSTS.hash else None
        self.face_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM face;""")} if CONSTS.face else None
        print('Finished.')

    def process_images(self, image_paths: list[str]) -> list[tuple]:
        features = []
        batch_features = []

        for image_path in image_paths:
            fs_img = FSProcessor(image_path)

            image_id = self.sha256_digest_to_image_id.get(fs_img.sha256_digest, None)

            feature = {}
            if self.exif_processor and (image_id not in self.exif_image_ids):
                fs_img.process()
                feature['exif'] = self.exif_processor.process(fs_img.img)

            if self.ocr_processor and (image_id not in self.ocr_image_ids):
                fs_img.process()
                feature['ocr'] = self.ocr_processor.process(image_path)

            if self.clip_processor and (image_id not in self.clip_image_ids):
                fs_img.process()
                batch_features.append(fs_img)

            if self.hash_processor and (image_id not in self.hash_image_ids):
                fs_img.process()
                feature['hash'] = self.hash_processor.process(fs_img.img)

            if self.face_processor and (image_id not in self.face_image_ids):
                fs_img.process()
                feature['face'] = self.face_processor.process(fs_img.img, fs_img.filename_secure)
            
            features.append((image_id, fs_img, feature))

        if self.clip_processor and batch_features:
            clip_features = self.clip_processor.process([fs_img.img for fs_img in batch_features])
            for i, feature in enumerate(features):
                features[i][2]['clip'] = clip_features[i]

        return features


def process_image_worker(image_path: str, processor: ImageProcessor):
    try:
        image_id, fs_img, features = processor.process_image(image_path)
        return (image_id, fs_img, features)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def write_to_db(processed_results: list[tuple]):
    with sqlite3.connect(CONSTS.db_path) as conn:
        cursor = conn.cursor()
        for p in processed_results:
            store_features_in_db(cursor, *p)
        conn.commit()
    processed_results.clear()


def load_images_and_store_in_db(root_image_folder: str, image_processor: ImageProcessor):
    file_paths = get_image_paths(root_image_folder)
    files_found = count_image_files(root_image_folder)
    max_files_to_process = min(files_found, CONSTS.max_files_to_process) if CONSTS.max_files_to_process else files_found

    processed_results = []

    count = 0
    batch_size = CONSTS.batch_size
    with tqdm.tqdm(total=max_files_to_process, position=0, desc='queue') as pbar:
        for path_batch in batched(file_paths, batch_size):
            if count > max_files_to_process:
                break

            processed_results.extend(image_processor.process_images(path_batch))

            if len(processed_results) >= batch_size * 2:
                write_to_db(processed_results)

            count += len(path_batch)
            pbar.update(len(path_batch))

    if processed_results:
        write_to_db(processed_results)


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

    for process in processor_types:
        if features.get(process) is not None:
            insert_feature(cursor, image_id, process, features[process])


def get_image_paths(root_dir) -> Generator:
    return (str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in valid_extensions)


if __name__ == '__main__':
    # set_start_method('spawn') # allow GPU multiprocessing
    init_db_all()

    ocr_processor = OCRProcessor(CONSTS.ocr_type) if CONSTS.ocr else None
    clip_processor = CLIPProcessor() if CONSTS.clip else None
    exif_processor = EXIFProcessor() if CONSTS.exif else None
    hash_processor = HashProcessor() if CONSTS.hash else None
    face_processor = FaceProcessor() if CONSTS.face else None

    image_processor = ImageProcessor(ocr_processor, clip_processor, exif_processor, hash_processor, face_processor)

    load_images_and_store_in_db(CONSTS.root_image_folder, image_processor)
