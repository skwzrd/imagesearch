import os
from enum import StrEnum
from functools import cache

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

from consts import CONSTS
from db import query_db
from db_api import get_sql_cols_from_d, get_sql_markers_from_d
from forms import SearchForm
from search import CLIPSearch, search_images
from utils import Perf, get_current_datetime, get_current_datetime_w_us_str


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


@cache
def get_image_count():
    return query_db('select count(*) as total_records from image;', one=True).total_records


@app.route('/', methods=['GET', 'POST'])
def index():
    form: SearchForm = SearchForm()
    results = []
    time_elapsed = None

    if form.validate_on_submit():
        form_fields = {}
        for field in CONSTS.form_fields:
            form_fields[field] = getattr(form, field).data

        file: FileStorage = form_fields.get('file')
        file_types: list[str] = form_fields.get('file_types')
        search_average_hash: bool = form_fields.get('search_average_hash')
        search_colorhash: bool = form_fields.get('search_colorhash')
        search_crop_resistant_hash: bool = form_fields.get('search_crop_resistant_hash')
        clip_file: bool = form_fields.get('clip_file')
        clip_text: str = form_fields.get('clip_text')
        exif_text: str = form_fields.get('exif_text')
        ocr_text: str = form_fields.get('ocr_text')
        min_face_count: int = form_fields.get('min_face_count')
        max_face_count: int = form_fields.get('max_face_count')
        noise: str = form_fields.get('noise')

        img = None
        if file:
            filepath = save_search(None, None, file)
            img = Image.open(filepath)

        p = Perf()
        results = search_images(
            img,
            clip_search=clip_search,
            clip_text=clip_text,
            clip_file=clip_file,
            exif_text=exif_text,
            ocr_text=ocr_text,
            min_face_count=min_face_count,
            max_face_count=max_face_count,
            noise=noise,
            search_average_hash=search_average_hash,
            search_colorhash=search_colorhash,
            search_crop_resistant_hash=search_crop_resistant_hash,
            file_types=file_types,
        )
        time_elapsed = round(p.check(), 3)

        if not results:
            flash('No results found', 'warning')

        form.data.clear()

    return render_template(
        'index.html',
        total_records=get_image_count(),
        time_elapsed=time_elapsed,
        form=form,
        results=results,
        search_result_limit=CONSTS.search_result_limit,
        form_fields=CONSTS.form_fields
    )


@app.route('/serve/<path:filename>')
def serve(filename: str):
    file_path = os.path.abspath(os.path.join('/', filename))
    if file_path.startswith(CONSTS.root_image_folder_web) and os.path.isfile(file_path):
        return send_file(file_path)
    abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)

