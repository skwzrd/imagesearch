import pickle
from enum import Enum
from functools import cache
from typing import NamedTuple

import torch
from imagehash import ImageHash, average_hash, colorhash, crop_resistant_hash
from PIL import Image

import clip
from configs import CONSTS
from db_api import get_sql_markers_from_d, query_db
from utils import sort_two_lists


def get_records_from_image_ids(image_ids):
    sql_string = f"""
    SELECT
        image.image_id,
        (image.filepath || '/' || image.filename_original) as path,
        exif.model,
        exif.ImageDescription,
        exif.UserComment,
        COALESCE(ocr.ocr_text, '') as ocr_text
    FROM image
        LEFT JOIN exif USING(image_id)
        LEFT JOIN clip USING(image_id)
        LEFT JOIN ocr USING(image_id)
        LEFT JOIN face USING(image_id)
        LEFT JOIN hash USING(image_id)
    WHERE image.image_id in ({get_sql_markers_from_d(image_ids)})
    ;"""
    return query_db(sql_string, args=image_ids)


class HashType(Enum):
    average_hash = 1
    colorhash = 2
    crop_resistant_hash = 3


class Hash(NamedTuple):
    image_id: int
    average_hash: ImageHash
    colorhash: ImageHash
    crop_resistant_hash: ImageHash


@cache
def get_image_hashes_from_db():
    # doesn't scale well currently, so we LIMIT
    sql_string = """
        SELECT
            image_id,
            average_hash,
            colorhash,
            crop_resistant_hash
        FROM hash
        LIMIT 20000
    ;"""
    rows = query_db(sql_string)

    hashes: list[Hash] = []
    for row in rows:
        h = Hash(
            row.image_id,
            pickle.loads(row.average_hash) if row.average_hash else None,
            pickle.loads(row.colorhash) if row.colorhash else None,
            pickle.loads(row.crop_resistant_hash) if row.crop_resistant_hash else None,
        )
        hashes.append(h)
    return hashes


class HashSearch:
    @staticmethod
    def search(img: Image, hash_type: HashType, max_hamming_distance: int=6, skip_image_ids: set[int]=None) -> set[int]:

        img_average_hash = average_hash(img) if hash_type == HashType.average_hash else None
        img_colorhash = colorhash(img) if hash_type == HashType.colorhash else None
        img_crop_resistant_hash = crop_resistant_hash(img) if hash_type == HashType.crop_resistant_hash else None

        hashes: list[Hash] = get_image_hashes_from_db()
        image_ids: set[int] = set()

        for hash in hashes:
            if skip_image_ids and hash.image_id in skip_image_ids:
                continue

            if hash_type == HashType.average_hash:
                hamming = hash.average_hash - img_average_hash
                if hamming <= max_hamming_distance:
                    image_ids.add(hash.image_id)
                    continue
            if hash_type == HashType.colorhash:
                hamming = hash.colorhash - img_colorhash
                if hamming <= max_hamming_distance:
                    image_ids.add(hash.image_id)
                    continue
            if hash_type == HashType.crop_resistant_hash:
                hamming = hash.crop_resistant_hash - img_crop_resistant_hash
                if hamming <= max_hamming_distance:
                    image_ids.add(hash.image_id)
                    continue

        return image_ids


@cache
def get_image_features_from_db():
    # doesn't scale well currently, so we LIMIT
    sql_string = """
        SELECT
            image_id,
            features
        FROM clip
        LIMIT 20000
    ;"""
    rows = query_db(sql_string)

    image_ids = [row.image_id for row in rows]
    features = [torch.tensor(pickle.loads(row.features)).to(CONSTS.device) for row in rows]

    return image_ids, torch.stack(features).squeeze()


class CLIPSearch:
    def __init__(self):
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('Finished!')


    def search_with_text(self, query: str, skip_image_ids: set[int]=None):
        text = clip.tokenize([query]).to(CONSTS.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text).float()
            return self.get_search_results(text_features, skip_image_ids=skip_image_ids)


    def search_with_image(self, img: Image, skip_image_ids: set[int]=None):
        query_image = self.preprocess(img).unsqueeze(0).to(CONSTS.device)
        with torch.no_grad():
            image_features = self.model.encode_image(query_image).float()
            return self.get_search_results(image_features, skip_image_ids=skip_image_ids)


    def get_search_results(self, input_features, skip_image_ids: set[int]=None):
        image_ids, features = get_image_features_from_db()

        if skip_image_ids:
            filtered_indices = [i for i, image_id in enumerate(image_ids) if (image_id not in skip_image_ids)]
            image_ids = [image_ids[i] for i in filtered_indices]
            features = features[filtered_indices]

        similarities = (input_features @ features.T).squeeze(0)
        n_indices = similarities.topk(CONSTS.search_result_limit).indices

        top_scores = [int(s) for s in similarities[n_indices].cpu().numpy()]
        top_image_ids = [image_ids[i] for i in n_indices]

        top_scores_filtered = []
        top_image_ids_filtered = []
        for score, image_id in zip(top_scores, top_image_ids):
            if score >= CONSTS.search_clip_match_threshold:
                top_scores_filtered.append(score)
                top_image_ids_filtered.append(image_id)

        top_scores, top_image_ids = sort_two_lists(top_scores_filtered, top_image_ids_filtered)

        drows = {row.image_id: row for row in get_records_from_image_ids(top_image_ids)}

        assert len(top_scores) == len(top_image_ids) == len(drows)

        results = []
        for i, image_id in enumerate(top_image_ids):
            results.append(dict(score=top_scores[i], **drows[image_id]))

        return results

def search_images(
    img: Image,
    clip_search: CLIPSearch = None,
    clip_text: str = None,
    clip_file: bool = None,
    exif_text: str = None,
    ocr_text: str = None,
    min_face_count: int = None,
    search_average_hash: bool = None,
    search_colorhash: bool = None,
    search_crop_resistant_hash: bool = None,
):
    conditions = []
    params = []
    image_ids = set()
    filtered = False

    if search_average_hash:
        ids = HashSearch.search(img, HashType.average_hash, max_hamming_distance=5, skip_image_ids=image_ids)
        image_ids = image_ids.intersection(ids) if filtered else ids
        if not image_ids: return []
        filtered = True

    if search_colorhash:
        ids = HashSearch.search(img, HashType.colorhash, max_hamming_distance=3, skip_image_ids=image_ids)
        image_ids = image_ids.intersection(ids) if filtered else ids
        if not image_ids: return []
        filtered = True

    if search_crop_resistant_hash:
        ids = HashSearch.search(img, HashType.crop_resistant_hash, max_hamming_distance=1, skip_image_ids=image_ids)
        image_ids = image_ids.intersection(ids) if filtered else ids
        if not image_ids: return []
        filtered = True

    if clip_text:
        clip_results = clip_search.search_with_text(clip_text, skip_image_ids=image_ids)
        ids = set([result['image_id'] for result in clip_results])
        image_ids = image_ids.intersection(ids) if filtered else ids
        if not image_ids: return []
        filtered = True
    
    if clip_file:
        clip_results = clip_search.search_with_image(img, skip_image_ids=image_ids)
        ids = set([result['image_id'] for result in clip_results])
        image_ids = image_ids.intersection(ids) if filtered else ids
        if not image_ids: return []
        filtered = True

    if len(image_ids) > 0:
        conditions.append(f"image.image_id IN ({','.join(['?'] * len(image_ids))})")
        params.extend(image_ids)

    if exif_text:
        conditions.append("(exif.ImageDescription LIKE ? OR exif.UserComment LIKE ?)")
        exif_term = f"%{exif_text}%"
        params.extend([exif_term, exif_term])

    if ocr_text:
        conditions.append("ocr.ocr_text LIKE ?")
        params.append(f"%{ocr_text}%")

    if min_face_count:
        conditions.append("face.face_count >= ?")
        params.append(min_face_count)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql_string = f"""
    SELECT
        image.image_id,
        (image.filepath || '/' || image.filename_original) as path,
        exif.model,
        exif.ImageDescription,
        exif.UserComment,
        COALESCE(ocr.ocr_text, '') as ocr_text,
        face.face_count
    FROM image
        LEFT JOIN exif USING(image_id)
        LEFT JOIN clip USING(image_id)
        LEFT JOIN ocr USING(image_id)
        LEFT JOIN face USING(image_id)
        LEFT JOIN hash USING(image_id)
    {where_clause}
    LIMIT {CONSTS.search_result_limit}
    ;"""

    rows = query_db(sql_string, params)
    return [dict(row) for row in rows]