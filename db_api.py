from functools import cache
from string import ascii_letters, digits
from typing import List

from PIL import ExifTags

from db import query_db


@cache
def get_exif_tag_d() -> dict:
    tags: dict = ExifTags.TAGS
    tags.update(ExifTags.GPSTAGS)
    for num, name in tags.items():
        tags[num] = name.replace('/', '')
        if set(tags[num]).difference(set(ascii_letters + digits)):
            raise ValueError(f'{tags[num]=}')
    assert len(tags.values()) > 200
    return tags


@cache
def get_exif_tag_names() -> List[str]:
    tags = list(set(get_exif_tag_d().values()))
    tags.sort(key=lambda x: x.lower())
    return tags


def get_sql_cols_from_d(d):
    return '`' + "`, `".join(d.keys()) + '`'


def get_sql_markers_from_d(d):
    return ", ".join(["?"] * len(d))


def init_table_exif():
    """For now, every EXIF tag is given a string type.
    """
    cols = get_exif_tag_names()

    cols = [col + ' TEXT' for col in cols]
    cols = ', '.join(cols)

    sql_string = f"""
        CREATE TABLE IF NOT EXISTS exif (
            exif_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            {cols},
            FOREIGN KEY(image_id) REFERENCES image(image_id) ON DELETE CASCADE
    );"""
    query_db(sql_string, commit=True)


def init_table_clip():
    sql_string = """
        CREATE TABLE IF NOT EXISTS clip (
            clip_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            features BLOB,
            FOREIGN KEY(image_id) REFERENCES image(image_id) ON DELETE CASCADE
        )
    ;"""
    query_db(sql_string, commit=True)


def init_table_ocr():
    sql_string = """
        CREATE TABLE IF NOT EXISTS ocr (
            ocr_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            ocr_text TEXT,
            FOREIGN KEY(image_id) REFERENCES image(image_id) ON DELETE CASCADE
        )
    ;"""
    query_db(sql_string, commit=True)


def init_table_image():
    sql_string = """
        CREATE TABLE IF NOT EXISTS image (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_time TEXT,
            sha256_digest TEXT UNIQUE,
            filename_original TEXT,
            filename_secure TEXT,
            filepath TEXT,
            filesize_bytes INTEGER,
            filetype TEXT
        )
    ;"""
    query_db(sql_string, commit=True)


def init_table_search_log():
    sql_string = """
    CREATE TABLE IF NOT EXISTS search_log (
        search_log_id INTEGER PRIMARY KEY AUTOINCREMENT,

        query_text TEXT,
        query_filepath TEXT,
        current_datetime DATETIME,

        x_forwarded_for TEXT,
        remote_addr TEXT,
        referrer TEXT,
        content_md5 TEXT,
        origin TEXT,
        scheme TEXT,
        method TEXT,
        root_path TEXT,
        path TEXT,
        query_string TEXT,
        user_agent TEXT,
        x_forwarded_proto TEXT,
        x_forwarded_host TEXT,
        x_forwarded_prefix TEXT,
        host TEXT,
        connection TEXT,
        content_length INTEGER
    )
    ;"""
    query_db(sql_string, commit=True)


def init_db_all():
    init_table_image()
    init_table_exif()
    init_table_clip()
    init_table_ocr()
    init_table_search_log()
