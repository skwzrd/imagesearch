import os
from enum import StrEnum

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for
)
from flask_bootstrap import Bootstrap5
from PIL import Image
from werkzeug.datastructures.file_storage import FileStorage
from werkzeug.utils import secure_filename

from configs import CONSTS
from db import query_db
from db_api import get_sql_cols_from_d, get_sql_markers_from_d
from forms import SearchForm
from search import CLIPSearch, search_images
from utils import get_current_datetime, get_current_datetime_w_us_str


class SearchType(StrEnum):
    clip_text = 'clip_text'
    clip_file = 'clip_file'
    exif_text = 'exif_text'
    ocr_text = 'ocr_text'
    face_count = 'face_count'
    average_hash_file = 'average_hash_file'


def basename(path):
    return os.path.basename(path)


def create_app():
    app = Flask(__name__)
    app.secret_key = CONSTS.flask_secret
    app.config['UPLOAD_FOLDER'] = CONSTS.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = CONSTS.MAX_CONTENT_LENGTH
    app.jinja_env.filters['basename'] = basename
    app.jinja_env.keep_trailing_newline = False
    app.jinja_env.trim_blocks = True

    Bootstrap5(app)

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

    return app


app = create_app()
clip_search: CLIPSearch = CLIPSearch()


def save_search(search_type: SearchType, text: str|None, file: FileStorage|None) -> str:
    if not file and not text:
        return

    if file:
        unique_prefix = get_current_datetime_w_us_str()
        filename_secure = f'{unique_prefix}__{secure_filename(file.filename)}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
        with open(filepath, 'wb') as f:
            file_data = file.read()
            f.write(file_data)
        os.chmod(filepath, 0o644) # o+r+w g+r a+r, no +x
    else:
        filepath = None

    d = dict(
        search_type=search_type,
        query_text=text,
        query_filepath=filepath,
        current_datetime=get_current_datetime(),

        x_forwarded_for=request.headers.get("X-Forwarded-For", None),
        remote_addr=request.headers.get("Remote-Addr", None),
        referrer=request.referrer,
        content_md5=request.content_md5,
        origin=request.origin,
        scheme=request.scheme,
        method=request.method,
        root_path=request.root_path,
        path=request.path,
        query_string=request.query_string.decode(),
        user_agent=request.user_agent.__str__(),
        x_forwarded_proto=request.headers.get("X-Forwarded-Proto", None),
        x_forwarded_host=request.headers.get("X-Forwarded-Host", None),
        x_forwarded_prefix=request.headers.get("X-Forwarded-Prefix", None),
        host=request.headers.get("Host", None),
        connection=request.headers.get("Connection", None),
        content_length=request.content_length,
    )

    sql_string = f"""INSERT INTO search_log ({get_sql_cols_from_d(d)}) VALUES ({get_sql_markers_from_d(d)});"""
    query_db(sql_string, args=list(d.values()), commit=True)

    return filepath


@app.route('/', methods=['GET', 'POST'])
def index():
    form: SearchForm = SearchForm()
    results = []

    if form.validate_on_submit():
        form_fields = {}
        for field in CONSTS.form_fields:
            form_fields[field] = getattr(form, field).data

        file: FileStorage = form_fields.get('file')
        search_average_hash: bool = form_fields.get('search_average_hash')
        search_colorhash: bool = form_fields.get('search_colorhash')
        search_crop_resistant_hash: bool = form_fields.get('search_crop_resistant_hash')
        clip_file: bool = form_fields.get('clip_file')
        clip_text: str = form_fields.get('clip_text')
        exif_text: str = form_fields.get('exif_text')
        ocr_text: str = form_fields.get('ocr_text')
        min_face_count: int = form_fields.get('min_face_count')
        max_face_count: int = form_fields.get('max_face_count')

        img = None
        if file:
            filepath = save_search(None, None, file)
            img = Image.open(filepath)

        results = search_images(
            img,
            clip_search=clip_search,
            clip_text=clip_text,
            clip_file=clip_file,
            exif_text=exif_text,
            ocr_text=ocr_text,
            min_face_count=min_face_count,
            max_face_count=max_face_count,
            search_average_hash=search_average_hash,
            search_colorhash=search_colorhash,
            search_crop_resistant_hash=search_crop_resistant_hash,
        )

        if not results:
            flash('No results found', 'warning')

        form.data.clear()

    return render_template('index.html', form=form, results=results, search_result_limit=CONSTS.search_result_limit, form_fields=CONSTS.form_fields)


@app.route('/serve/<path:filename>')
def serve(filename: str):
    file_path = os.path.abspath(os.path.join('/', filename))
    if file_path.startswith(CONSTS.root_image_folder) and os.path.isfile(file_path):
        return send_file(file_path)
    abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)

