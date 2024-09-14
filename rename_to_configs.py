class CONSTS:

    # processing and web search
    image_dir = '/home/USER/Documents/images/images'
    upload_folder = '/home/USER/Documents/code/clip/static/uploads'
    db_path = '/home/USER/Documents/code/clip/latest.db'

    valid_extensions = (".png", ".jpg", ".jpeg") # only these have been tested

    # processing
    device = "cuda" # "cpu"
    ocr_type = "ocrs" # "doctr" "tesseract" # pick one. # does not apply if CONSTS.ocr = 0 | False.

    # 1 - True, 0 - False
    exif = 1
    clip = 1
    ocr = 1

    limit = 0 # N images to process. If 0, no limit.

    # web search
    n_results = 3 # search result count
    secret = '3456789yefuenfhwr78ty34gtfhot3hjpo34ti8765' # change me