import pickle
from functools import cache

import torch
from PIL import Image

import clip
from configs import CONSTS
from db_api import query_db


class CLIPSearch:
    def __init__(self):
        print('Loading CLIP Model...')
        self.model, self.preprocess = clip.load("ViT-B/32", device=CONSTS.device)
        print('Finished!')


    @cache
    def get_image_features_from_db(self):
        data = query_db("""
            SELECT
                (image.filepath || '/' || image.filename_original) as path,
                clip.features,
                exif.model as exif_summary,
                ocr.ocr_text
            FROM image
                left join exif USING(image_id)
                left join clip USING(image_id)
                left join ocr USING(image_id)
        ;""")
        paths = []
        features = []
        exif_summaries = []
        ocr_texts = []
        for row in data:
            if row.features:
                paths.append(row.path)
                features.append(torch.tensor(pickle.loads(row.features)).to(CONSTS.device))
                exif_summaries.append(row.exif_summary)
                ocr_texts.append(row.ocr_text)
            else:
                raise ValueError(f"{row.features=}")
        return paths, torch.stack(features).squeeze(), exif_summaries, ocr_texts


    def search_with_text(self, query):
        text = clip.tokenize([query]).to(CONSTS.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text).float()
            return self.get_n_results(text_features)


    def search_with_image(self, query_image_path):
        query_image = self.preprocess(Image.open(query_image_path)).unsqueeze(0).to(CONSTS.device)
        with torch.no_grad():
            image_features = self.model.encode_image(query_image).float()
            return self.get_n_results(image_features)


    def get_n_results(self, input_features):
            # doesn't scale well currently
            paths, features, exif_summaries, ocr_texts = self.get_image_features_from_db()

            similarities = (input_features @ features.T).squeeze(0)
            n_indices = similarities.topk(CONSTS.n_results).indices
            n_results = [
                (
                    paths[idx],
                    similarities[idx].cpu().item(),
                    exif_summaries[idx],
                    ocr_texts[idx]
                )
                for idx in n_indices
            ]
            print(n_results)
            return n_results
