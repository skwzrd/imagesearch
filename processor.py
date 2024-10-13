import os
import pickle
from collections import OrderedDict
from typing import Dict, List

import torch
from numpy import array, ndarray
from PIL import Image
from werkzeug.utils import secure_filename

from consts import CONSTS
from db import query_db
from utils import get_exif_tag_d, get_sha256

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

if CONSTS.ski:
    from numpy import nanmean, all as nanall, isnan
    from skimage.restoration import estimate_sigma


class SkiProcessor:
    def __init__(self) -> None:
        pass

    def process(self, img_array: ndarray) -> dict:
        """
        0: high noise, infographs, images not take by a camera
        0.2 and below: should remove all garbage, cursed images
        0.2 and up: real photos
        """
        noises = estimate_sigma(img_array, channel_axis=-1, average_sigmas=False)
        
        average_noise = nanmean(average_noise) if not nanall(isnan(average_noise)) else 0.0

        noise_1, noise_2, noise_3 = (noises + [None, None, None])[:3]

        return dict(
            noise=round(average_noise, 4),
            noise_1=round(noise_1, 4) if noise_1 is not None else None,
            noise_2=round(noise_2, 4) if noise_2 is not None else None,
            noise_3=round(noise_3, 4) if noise_3 is not None else None,
        )


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
        self.sha256_digest: str = get_sha256(image_path)

        self.filename_original: str = None
        self.filepath: str = None
        self.filesize_bytes: int = None
        self.img: Image = None
        self.filetype: str = None
        self.img_array: ndarray = None

        self.processed = False

    def process(self, img_array: bool=False):
        """We may not need to process all of this when our object is first instantiated."""
        if not self.processed:
            self.filename_original: str = os.path.basename(self.image_path)
            self.filepath: str = os.path.dirname(self.image_path)
            self.filesize_bytes: int = os.path.getsize(self.image_path)
            self.img: Image = Image.open(self.image_path)
            self.filetype: str = self.img.format.lower()
            
            self.processed = True

        if img_array and self.img_array is None:
            self.img_array = array(self.img.convert('RGB'))


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
        model = 'hog' if CONSTS.device == 'cuda' else 'hog'
        face_locations = face_recognition.face_locations(fs_img.img_array, model=model) # 1-2 images/s

        face_count = len(face_locations)
        face_encodings = None

        if face_count:
            if CONSTS.face_save:
                self.save_images(fs_img.img_array, face_locations, secure_filename(fs_img.filename_original))

            if CONSTS.face_encodings:
                face_encodings: List[array] = face_recognition.face_encodings(fs_img.img_array, face_locations)

        face_encodings = pickle.dumps(face_encodings)

        return dict(face_count=face_count, face_encodings=face_encodings)


class CLIPProcessor:
    def __init__(self) -> None:
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)

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


class ImageProcessor:
    def __init__(self, ocr_processor=None, clip_processor=None, exif_processor=None, hash_processor=None, face_processor=None, ski_processor=None):
        self.ocr_processor: OCRProcessor = ocr_processor
        self.clip_processor: CLIPProcessor = clip_processor
        self.exif_processor: EXIFProcessor = exif_processor
        self.hash_processor: HashProcessor = hash_processor
        self.face_processor: FaceProcessor = face_processor
        self.ski_processor: SkiProcessor = ski_processor

        print('Setting up sets...')
        self.sha256_digest_to_image_id: dict = {row.sha256_digest: row.image_id for row in query_db("""SELECT image_id, sha256_digest FROM image;""")}

        self.clip_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM clip;""")} if CONSTS.clip else None
        self.exif_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM exif;""")} if CONSTS.exif else None
        self.ocr_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM ocr;""")} if CONSTS.ocr else None
        self.hash_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM hash;""")} if CONSTS.hash else None
        self.face_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM face;""")} if CONSTS.face else None
        self.ski_image_ids: set = {row.image_id for row in query_db("""SELECT image_id FROM ski;""")} if CONSTS.ski else None

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
            fs_img.process(img_array=True)
            features['face'] = self.face_processor.process(fs_img)

        if self.ski_processor and (image_id not in self.ski_image_ids):
            fs_img.process(img_array=True)
            features['ski'] = self.ski_processor.process(fs_img.img_array)

        if not features:
            return

        return image_id, fs_img, features