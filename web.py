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
        file: FileStorage = form.file.data
        search_average_hash: bool = form.search_average_hash.data
        search_colorhash: bool = form.search_colorhash.data
        search_crop_resistant_hash: bool = form.search_crop_resistant_hash.data
        clip_file: bool = form.clip_file.data
        clip_text: str = form.clip_text.data
        exif_text: str = form.exif_text.data
        ocr_text: str = form.ocr_text.data
        min_face_count: int = form.min_face_count.data

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
            search_average_hash=search_average_hash,
            search_colorhash=search_colorhash,
            search_crop_resistant_hash=search_crop_resistant_hash,
        )

        if not results:
            flash('No results found', 'warning')

        form.data.clear()

    return render_template('index.html', form=form, results=results, search_result_limit=CONSTS.search_result_limit)


@app.route('/serve/<path:filename>')
def serve(filename: str):
    file_path = os.path.abspath(os.path.join('/', filename))
    if file_path.startswith(CONSTS.root_image_folder) and os.path.isfile(file_path):
        return send_file(file_path)
    abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)

