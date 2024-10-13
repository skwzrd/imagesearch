class CONSTS:
    root_image_folder_web = '/home/USER' # any paths from this dir can be served
    root_image_folder_processors = '/home/USER/Desktop/Documents/images/images' # which dir to process

    db_path = '/home/USER/Documents/code/clip/00000.db'
    device = 'cuda' # 'cpu' 'cuda'

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

    ski = 1 # 0 - this is allows for a nice filter that removes non-photos, and low res images

    # general
    max_files_to_process = 0 # 0 == all files


    # web
    UPLOAD_FOLDER = '/home/USER/Documents/code/clip/static/uploads'
    flask_secret = 'eererer36eyher4y346t4tg4t4ef' # change me
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 # 10 MB
    search_result_limit = 20
    search_clip_match_threshold = 25 # 30 is OK, 135 seems to be the max.
    search_hd_limit_average_hash = 2 # 0-1 is identical/very similar, 5 is sorta similar, 6+ is not very similar
    search_hd_limit_colorhash = 5 # (identical/similar) 0 <---> 20+ (dissimilar)
    search_hd_limit_crop_resistant_hash = 2 # (identical/similar) 0 <---> 20+ (dissimilar)
