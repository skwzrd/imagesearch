import sqlite3
from multiprocessing import set_start_method
from pathlib import Path
from typing import Generator

import tqdm

from consts import CONSTS, valid_extensions
from db_api import init_db_all, store_features_in_db
from processor import (
    CLIPProcessor,
    EXIFProcessor,
    FaceProcessor,
    HashProcessor,
    ImageProcessor,
    OCRProcessor,
    SkiProcessor
)
from utils import count_image_files


def get_image_paths(root_dir) -> Generator:
    return (str(p) for p in Path(root_dir).rglob('*') if p.suffix.lower() in valid_extensions)


def load_images_and_store_in_db(root_image_folder: str, image_processor: ImageProcessor):
    file_paths = get_image_paths(root_image_folder)
    files_found = count_image_files(root_image_folder)
    max_files_to_process = min(files_found, CONSTS.max_files_to_process) if CONSTS.max_files_to_process else files_found

    conn = sqlite3.connect(CONSTS.db_path)
    cursor = conn.cursor()
    batch_commit = 100
    batch_commit_i = 0

    try:
        for i, path in tqdm.tqdm(enumerate(file_paths), total=max_files_to_process, desc='progress'):
            if i >= max_files_to_process:
                break

            result = image_processor.process_image(path)
            if result:
                image_id, fs_img, features = result
                store_features_in_db(cursor, image_id, fs_img, features)

            if batch_commit_i > batch_commit:
                conn.commit()
                batch_commit_i = 0

            batch_commit_i += 1
    finally:
        conn.commit()
        cursor.close()
        conn.close()


if __name__ == '__main__':
    set_start_method('spawn') # allow GPU multiprocessing
    init_db_all()

    ocr_processor = OCRProcessor(CONSTS.ocr_type) if CONSTS.ocr else None
    clip_processor = CLIPProcessor() if CONSTS.clip else None
    exif_processor = EXIFProcessor() if CONSTS.exif else None
    hash_processor = HashProcessor() if CONSTS.hash else None
    face_processor = FaceProcessor() if CONSTS.face else None
    ski_processor = SkiProcessor() if CONSTS.ski else None

    image_processor = ImageProcessor(ocr_processor, clip_processor, exif_processor, hash_processor, face_processor, ski_processor)

    load_images_and_store_in_db(CONSTS.root_image_folder_processors, image_processor)