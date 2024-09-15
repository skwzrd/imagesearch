class CONSTS:

    # processors and web
    root_image_folder = '/home/USER/Documents/images/images'
    db_path = '/home/USER/Documents/code/clip/latest.db'
    device = 'cuda' # 'cpu'

    # processor type toggles
    exif = 1 # 0
    clip = 1 # 0
    ocr = 1 # 0

    processor_file_limit = 0 # 0 == all files
    ocr_type = 'ocrs' # 'doctr' 'tesseract'

    # web
    UPLOAD_FOLDER = '/home/USER/Documents/code/clip/static/uploads'
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 # 10 MB
    search_result_limit = 4
    search_clip_match_threshold = 30.0 # 30 is OK, 135 seems to be the max.
    flask_secret = 'eererer3545t4tg4t4ef' # change me
