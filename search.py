import pickle
from functools import cache

import torch
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
    WHERE image.image_id in ({get_sql_markers_from_d(image_ids)})
    ;"""
    return query_db(sql_string, args=image_ids)


class CLIPSearch:
    def __init__(self):
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('Finished!')


    @cache
    def get_image_features_from_db(self):
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


    def search_with_text(self, query):
        text = clip.tokenize([query]).to(CONSTS.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text).float()
            return self.get_search_results(text_features)


    def search_with_image(self, query_image_path):
        query_image = self.preprocess(Image.open(query_image_path)).unsqueeze(0).to(CONSTS.device)
        with torch.no_grad():
            image_features = self.model.encode_image(query_image).float()
            return self.get_search_results(image_features)


    def get_search_results(self, input_features):
        image_ids, features = self.get_image_features_from_db()

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


def search_exif(text):
    search_term = f"%{text}%"
    rows = query_db("""SELECT image_id FROM exif WHERE ImageDescription LIKE ? or UserComment LIKE ? LIMIT ?;""", (search_term, search_term, CONSTS.search_result_limit))
    return get_records_from_image_ids([row.image_id for row  in rows])


def search_ocr(text):
    search_term = f"%{text}%"
    rows = query_db("""SELECT image_id FROM ocr WHERE ocr_text LIKE ? LIMIT ?;""", (search_term, CONSTS.search_result_limit))
    return get_records_from_image_ids([row.image_id for row  in rows])

