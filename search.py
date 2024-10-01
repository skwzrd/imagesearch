import pickle
from collections import defaultdict
from enum import Enum, StrEnum
from functools import cache
from typing import NamedTuple

import torch
from imagehash import ImageHash, average_hash, colorhash, crop_resistant_hash
from PIL import Image

import clip
from consts import CONSTS
from db_api import query_db
from utils import sort_two_lists


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
    def search(img: Image, hash_type: HashType, max_hamming_distance: int=6, skip_image_ids: set[int]=None) -> dict[int, int]:

        img_average_hash = average_hash(img) if hash_type == HashType.average_hash else None
        img_colorhash = colorhash(img) if hash_type == HashType.colorhash else None
        img_crop_resistant_hash = crop_resistant_hash(img) if hash_type == HashType.crop_resistant_hash else None

        hashes: list[Hash] = get_image_hashes_from_db()
        image_ids_2_hamming = {}

        for hash in hashes:
            if skip_image_ids and hash.image_id in skip_image_ids:
                continue

            if hash_type == HashType.average_hash:
                hamming = hash.average_hash - img_average_hash
                if hamming <= max_hamming_distance:
                    image_ids_2_hamming[hash.image_id] = hamming
                    continue
            if hash_type == HashType.colorhash:
                hamming = hash.colorhash - img_colorhash
                if hamming <= max_hamming_distance:
                    image_ids_2_hamming[hash.image_id] = hamming
                    continue
            if hash_type == HashType.crop_resistant_hash:
                hamming = hash.crop_resistant_hash - img_crop_resistant_hash
                if hamming <= max_hamming_distance:
                    image_ids_2_hamming[hash.image_id] = hamming
                    continue

        return image_ids_2_hamming


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


    def get_search_results(self, input_features, skip_image_ids: set[int]=None) -> dict[int, int]:
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

        assert len(top_image_ids) == len(top_scores)

        return {image_id: score for image_id, score in zip(top_image_ids, top_scores)}


class Metrics(StrEnum):
    # printed to the browser
    FaceCount = 'FaceCount'
    AverageHash = 'AverageHash'
    ColorHash = 'ColorHash'
    CropResistantHash = 'CropResistantHash'
    ClipText = 'ClipText'
    ClipFile = 'ClipFile'


def search_images(
    img: Image,
    clip_search: CLIPSearch = None,
    clip_text: str = None,
    clip_file: bool = None,
    exif_text: str = None,
    ocr_text: str = None,
    min_face_count: int = None,
    max_face_count: int = None,
    search_average_hash: bool = None,
    search_colorhash: bool = None,
    search_crop_resistant_hash: bool = None,
):
    conditions = []
    params = []
    image_ids = set()
    filtered = False
    image_ids_2_metrics = defaultdict(lambda: defaultdict(dict))

    def metric_routine(metric: str, ids_2_metric: dict[int, int]) -> bool:
        nonlocal filtered
        nonlocal image_ids
        nonlocal image_ids_2_metrics

        if filtered:
            image_ids = image_ids.intersection(ids_2_metric.keys())
        else:
            image_ids = set(ids_2_metric.keys()) # the first time we are filtering
        filtered = True

        if not image_ids:
            return False

        for image_id in image_ids:
            image_ids_2_metrics[image_id][metric] = ids_2_metric[image_id]

        return True

    if search_average_hash:
        ids_2_metric = HashSearch.search(img, HashType.average_hash, max_hamming_distance=CONSTS.search_hd_limit_average_hash)
        if not metric_routine(Metrics.AverageHash, ids_2_metric):
            return []
    
    if search_colorhash:
        ids_2_metric = HashSearch.search(img, HashType.colorhash, max_hamming_distance=CONSTS.search_hd_limit_colorhash)
        if not metric_routine(Metrics.ColorHash, ids_2_metric):
            return []

    if search_crop_resistant_hash:
        ids_2_metric = HashSearch.search(img, HashType.crop_resistant_hash, max_hamming_distance=CONSTS.search_hd_limit_crop_resistant_hash)
        if not metric_routine(Metrics.CropResistantHash, ids_2_metric):
            return []

    if clip_text:
        ids_2_metric = clip_search.search_with_text(clip_text)
        if not metric_routine(Metrics.ClipText, ids_2_metric):
            return []
    
    if clip_file:
        ids_2_metric = clip_search.search_with_image(img)
        if not metric_routine(Metrics.ClipFile, ids_2_metric):
            return []

    if len(image_ids) > 0:
        conditions.append(f"image.image_id IN ({','.join(['?'] * len(image_ids))})")
        params.extend(image_ids)

    if exif_text:
        conditions.append("(LOWER(exif.ImageDescription) LIKE ? OR LOWER(exif.UserComment) LIKE ?)")
        exif_term = f"%{exif_text.lower()}%"
        params.extend([exif_term, exif_term])

    if ocr_text:
        conditions.append("LOWER(ocr.ocr_text) LIKE ?")
        params.append(f"%{ocr_text.lower()}%")

    if min_face_count:
        conditions.append("face.face_count >= ?")
        params.append(min_face_count)

    if max_face_count:
        conditions.append("face.face_count <= ?")
        params.append(max_face_count)

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
    results = []
    for row in rows:
        image_ids_2_metrics[row.image_id][Metrics.FaceCount] = row.face_count
        metrics = image_ids_2_metrics.get(row.image_id, None)
        result = dict(
            combined_score=get_combined_score(metrics),
            metrics=metrics,
            **row,
        )
        results.append(result)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results


def get_combined_score(metrics: dict) -> int:
    """Creates a score where `(worst) 0 <---> 100 (best)`.
    """
    score = 0
    counts = 0
    for metric, value in metrics.items():
        if metric in {Metrics.AverageHash, Metrics.ColorHash, Metrics.CropResistantHash}:
            value = max(100 - (value ** 2), 0) # 0=100, 5=75, >10=0
            counts += 1
            score += value

        elif metric in {Metrics.ClipText, Metrics.ClipFile}:
            value = value / 1.35 # 135 is the largest value I've seen, so we normalize to it
            counts += 1
            score += value

    return min(int(score / (max(counts, 1))), 100)
