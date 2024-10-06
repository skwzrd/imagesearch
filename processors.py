import os
import pickle
import sqlite3
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import Manager, Process, Queue, set_start_method
from pathlib import Path
from queue import Empty
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
from utils import count_image_files, get_dt_format, get_sha256

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
            self.filepath: str = os.path.dirname(self.image_path)
            self.filesize_bytes: int = os.path.getsize(self.image_path)
            self.img: Image = Image.open(self.image_path)
            self.filetype: str = self.img.format.lower()
            
            self.processed = True


class FaceProcessor:
    def __init__(self) -> None:
        pass

    def save_images(self, img_array: list, face_locations: tuple[int], filename_secure: str):
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_image = img_array[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            save_dir = os.path.join(os.path.dirname(__file__), 'ignore')
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                os.chmod(save_dir, 0o777)
            pil_image.save(os.path.join(save_dir, f'{filename_secure}___{str(i).zfill(3)}.png'))

    def process(self, fs_img: FSProcessor) -> dict:
        img_array = array(fs_img.img.convert('RGB'))
        model = 'hog' if CONSTS.device == 'cuda' else 'hog'
        face_locations = face_recognition.face_locations(img_array, model=model) # 1-2 images/s

        face_count = len(face_locations)
        face_encodings = None

        if face_count:
            if CONSTS.face_save:
                self.save_images(img_array, face_locations, secure_filename(fs_img.filename_original))

            if CONSTS.face_encodings:
                face_encodings: List[array] = face_recognition.face_encodings(img_array, face_locations)

        face_encodings = pickle.dumps(face_encodings)

        return dict(face_count=face_count, face_encodings=face_encodings)


class CLIPProcessor:
    def __init__(self) -> None:
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('Finished')

    def process(self, img: Image) -> dict:
        image = self.preprocess(img).unsqueeze(0).to(CONSTS.device) # 0.08s - bottleneck 1
        with torch.no_grad():
            image_features = self.model.encode_image(image).float() # 0.03s - bottleneck 2
        return {'features': pickle.dumps(image_features.cpu().numpy())} # BLOB for column 'features'


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

    def process_image(self, image_path: str):
        fs_img = FSProcessor(image_path)

        image_id = self.sha256_digest_to_image_id.get(fs_img.sha256_digest, None)
        features = {}

        if self.exif_processor and (image_id not in self.exif_image_ids):
            fs_img.process()
            features['exif'] = self.exif_processor.process(fs_img.img)

        if self.ocr_processor and (image_id not in self.ocr_image_ids):
            fs_img.process()
            features['ocr'] = self.ocr_processor.process(image_path)

        if self.clip_processor and (image_id not in self.clip_image_ids):
            fs_img.process()
            features['clip'] = self.clip_processor.process(fs_img.img)

        if self.hash_processor and (image_id not in self.hash_image_ids):
            fs_img.process()
            features['hash'] = self.hash_processor.process(fs_img.img)

        if self.face_processor and (image_id not in self.face_image_ids):
            fs_img.process()
            features['face'] = self.face_processor.process(fs_img)

        return image_id, fs_img, features


def db_writer(write_queue: Queue, db_path: str, db_batch_size: int = 50, timeout: int = 5):
    """Put `None` into `write_queue` to complete process."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    batch = []

    with tqdm.tqdm(total=db_batch_size, desc="write queue", position=1, leave=False) as pbar:
        while True:
            try:
                task = write_queue.get(timeout=timeout)  # raises Empty after timeout
                if task is None:
                    break  # exit signal received

                batch.append(task)
                pbar.update(1)

                if len(batch) >= db_batch_size:
                    for image_id, fs_img, features in batch:
                        store_features_in_db(cursor, image_id, fs_img, features)
                        pbar.update(-1)
                    conn.commit()
                    batch.clear()
                    pbar.reset()

            except Empty:
                if batch:
                    for image_id, fs_img, features in batch:
                        store_features_in_db(cursor, image_id, fs_img, features)
                        pbar.update(-1)
                    conn.commit()
                    batch.clear()
                    pbar.reset()

    # commits any remaining data in the batch
    if batch:
        for image_id, fs_img, features in batch:
            store_features_in_db(cursor, image_id, fs_img, features)
            pbar.update(-1)
        conn.commit()
    batch.clear()
    pbar.reset()
    cursor.close()
    conn.close()
    print('\nAll records saved.')


def process_image_worker(image_path: str, processor: ImageProcessor, write_queue: Queue):
    try:
        image_id, fs_img, features = processor.process_image(image_path)
        write_queue.put((image_id, fs_img, features))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def load_images_and_store_in_db(root_image_folder: str, image_processor: ImageProcessor):
    file_paths = get_image_paths(root_image_folder)
    files_found = count_image_files(root_image_folder)
    max_files_to_process = min(files_found, CONSTS.max_files_to_process) if CONSTS.max_files_to_process else files_found

    manager = Manager()
    write_queue = manager.Queue() # share queue across processes

    db_process = Process(target=db_writer, args=(write_queue, CONSTS.db_path, CONSTS.db_batch_size))
    db_process.start()

    error_occurred = False

    with ProcessPoolExecutor(max_workers=CONSTS.max_workers) as executor:
        futures = []
        for path in file_paths:
            if len(futures) < max_files_to_process:
                futures.append(executor.submit(process_image_worker, path, image_processor, write_queue))

        with tqdm.tqdm(total=len(futures), position=0, desc='process queue') as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in future: {e}")
                    error_occurred = True
                    break
                finally:
                    pbar.update(1)

    if error_occurred:
        executor.shutdown(wait=False)
        print("Shutting down due to an error...")

    write_queue.put(None) # signal db_writer to terminate
    db_process.join()


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
    set_start_method('spawn') # allow GPU multiprocessing
    init_db_all()

    ocr_processor = OCRProcessor(CONSTS.ocr_type) if CONSTS.ocr else None
    clip_processor = CLIPProcessor() if CONSTS.clip else None
    exif_processor = EXIFProcessor() if CONSTS.exif else None
    hash_processor = HashProcessor() if CONSTS.hash else None
    face_processor = FaceProcessor() if CONSTS.face else None

    image_processor = ImageProcessor(ocr_processor, clip_processor, exif_processor, hash_processor, face_processor)

    load_images_and_store_in_db(CONSTS.root_image_folder, image_processor)