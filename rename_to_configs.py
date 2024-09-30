class CONSTS:
    # processors and web
    root_image_folder = '/home/USER/Documents/images/images'
    db_path = '/home/USER/Documents/code/clip/00000.db'
    device = 'cuda' # 'cpu'

    # processor type toggles
    exif = 1 # 0
    clip = 1 # 0
    ocr = 1 # 0

    hash_average = 1 # 0
    hash_color = 1 # 0
    hash_crop_resistant = 1 # 0

    face_count = 1     # 0 - do face detection
    face_encodings = 1 # 0 - save face encodings as BLOBs in db
    face_save = 1      # 0 - save faces found in images

    processor_file_limit = 0 # 0 == all files
    ocr_type = 'ocrs' # 'doctr' 'tesseract'

    # web
    UPLOAD_FOLDER = '/home/USER/Documents/code/clip/static/uploads'
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 # 10 MB
    search_result_limit = 100
    search_clip_match_threshold = 25.0 # 30 is OK, 135 seems to be the max.
    flask_secret = 'eererer36eyher4y346t4tg4t4ef' # change me

    # do not touch
    hash = any([hash_average, hash_color, hash_crop_resistant])
    face = any([face_count, face_encodings, face_save])
