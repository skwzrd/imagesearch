class CONSTS:
    # processors and web
    root_image_folder = '/home/USER/Documents/images/images'
    db_path = '/home/USER/Documents/code/clip/00000.db'
    device = 'cuda' # 'cpu'

    # processor type toggles: 1 = on, 0 = off
    exif = 1 # 0

    clip = 1 # 0

    ocr = 1 # 0
    ocr_type = 'ocrs' # 'doctr' 'tesseract'

    hash_average = 1 # 0
    hash_color = 1 # 0
    hash_crop_resistant = 1 # 0

    face_count = 1     # 0 - do face detection
    face_encodings = 1 # 0 - save face encodings as BLOBs in db
    face_save = 1      # 0 - save faces found in images

    # general
    max_files_to_process = 0 # 0 == all files
    db_batch_size = 8
    max_workers = 6

    # web
    UPLOAD_FOLDER = '/home/USER/Documents/code/clip/static/uploads'
    flask_secret = 'eererer36eyher4y346t4tg4t4ef' # change me
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 # 10 MB
    search_result_limit = 12
    search_clip_match_threshold = 25 # 30 is OK, 135 seems to be the max.
    search_hd_limit_average_hash = 2 # 0-1 is identical/very similar, 5 is sorta similar, 6+ is not very similar
    search_hd_limit_colorhash = 5 # (identical/similar) 0 <---> 20+ (dissimilar)
    search_hd_limit_crop_resistant_hash = 2 # (identical/similar) 0 <---> 20+ (dissimilar)
