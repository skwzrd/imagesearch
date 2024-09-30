from functools import cache
from string import ascii_letters, digits
from typing import List

from PIL import ExifTags

from consts import processor_types
from db import query_db


def set_sql_settings_optimize_bulk_writing():
    """`journal_mode` is the only setting that is altered. This needs studying."""
    sqls = [
        'PRAGMA journal_mode = WAL;', # Enables Write-Ahead Logging, which allows for faster writes by keeping changes in a separate WAL file until the database is checkpointed.
        'PRAGMA synchronous = OFF;', # Disables synchronous writes, making writes faster but increasing the risk of data loss if there's a crash or power failure.
        'PRAGMA cache_size = 100000;', # Increases the number of pages that can be cached in memory, which improves write performance by reducing disk I/O.
        'PRAGMA temp_store = MEMORY;', # Stores temporary tables and indexes in memory instead of on disk.
        'PRAGMA foreign_keys = OFF;', # Temporarily disables foreign key constraint checks, which can slow down bulk inserts.
        'PRAGMA page_size = 65536;', # Increases the size of database pages, meaning fewer disk I/O operations are required for large datasets.
        'PRAGMA auto_vacuum = NONE;', # Prevents automatic database vacuuming during bulk writes, which would otherwise slow down the process.
    ]
    for sql in sqls:
        query_db(sql, commit=True)


def print_sql_settings():
    sqls = [
        'PRAGMA journal_mode;',
        'PRAGMA synchronous;',
        'PRAGMA cache_size;',
        'PRAGMA temp_store;',
        'PRAGMA foreign_keys;',
        'PRAGMA page_size;',
        'PRAGMA auto_vacuum;',
    ]
    for sql in sqls:
        print(query_db(sql))
    print()


def set_sql_settings_default():
    sqls = [
        'PRAGMA journal_mode = DELETE;', # This reverts SQLite to use the rollback journal mode, which is the default and most compatible mode.
        'PRAGMA synchronous = FULL;', # This restores full write safety, ensuring that all writes are fully committed to disk, reducing the risk of corruption.
        'PRAGMA cache_size = -2000;', # By default, SQLite sets the cache size to about 2 MB (2000 KB). The negative value indicates size in kilobytes.
        'PRAGMA temp_store = DEFAULT;', # Reverts to storing temporary tables on disk (the default behavior).
        'PRAGMA foreign_keys = ON;', # Enables foreign key constraints, which is the default behavior since SQLite version 3.6.19.
        'PRAGMA page_size = 4096;', # The default page size is usually 4 KB (4096 bytes), though this can vary depending on the platform.
        'PRAGMA auto_vacuum = NONE;', # This setting doesn't need to be changed since the default is also NONE.
    ]
    for sql in sqls:
        query_db(sql, commit=True)


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


def init_table_hash():
    sql_string = """
        CREATE TABLE IF NOT EXISTS hash (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            average_hash BLOB DEFAULT NULL,
            colorhash BLOB DEFAULT NULL,
            crop_resistant_hash BLOB DEFAULT NULL,
            FOREIGN KEY(image_id) REFERENCES image(image_id) ON DELETE CASCADE
        )
    ;"""
    query_db(sql_string, commit=True)


def init_table_face():
    sql_string = """
        CREATE TABLE IF NOT EXISTS face (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_count TEXT,
            face_encodings BLOB,
            FOREIGN KEY(image_id) REFERENCES image(image_id) ON DELETE CASCADE
        )
    ;"""
    query_db(sql_string, commit=True)


def init_table_search_log():
    sql_string = """
    CREATE TABLE IF NOT EXISTS search_log (
        search_log_id INTEGER PRIMARY KEY AUTOINCREMENT,

        search_type TEXT,
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


def init_indexes():
    for processor_type in processor_types:
        sql = f"""CREATE INDEX IF NOT EXISTS idx_{processor_type}_image_id ON {processor_type}(image_id);"""
        query_db(sql, commit=True)


def init_db_all():
    init_table_image()
    init_table_exif()
    init_table_clip()
    init_table_ocr()
    init_table_hash()
    init_table_face()

    init_indexes()

    init_table_search_log()
