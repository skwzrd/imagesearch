from configs import CONSTS

valid_extensions = (".png", ".jpg", ".jpeg", ".gif")

clip_valid_extensions = (".png", ".jpg", ".jpeg", ".gif")
ocr_valid_extensions = (".png", ".jpg", ".jpeg")
exif_valid_extensions = (".png", ".jpg", ".jpeg")

processor_types = ['exif', 'hash', 'face', 'ocr', 'clip']

CONSTS.hash = any([CONSTS.hash_average, CONSTS.hash_color, CONSTS.hash_crop_resistant])
CONSTS.face = any([CONSTS.face_count, CONSTS.face_encodings, CONSTS.face_save])

form_fields = ['search', 'csrf_token']
if CONSTS.hash or CONSTS.clip: form_fields.append('file')
if CONSTS.hash_average: form_fields.append('search_average_hash')
if CONSTS.hash_color: form_fields.append('search_colorhash')
if CONSTS.hash_crop_resistant: form_fields.append('search_crop_resistant_hash')
if CONSTS.clip: form_fields.append('clip_file')
if CONSTS.clip: form_fields.append('clip_text')
if CONSTS.exif: form_fields.append('exif_text')
if CONSTS.ocr: form_fields.append('ocr_text')
if CONSTS.face: form_fields.append('min_face_count')
if CONSTS.face: form_fields.append('max_face_count')
CONSTS.form_fields = form_fields
