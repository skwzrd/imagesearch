import sqlite3
from contextlib import contextmanager

from configs import CONSTS


def query_db(query, args=(), one=False, commit=False):
    conn = get_db_conn()
    cursor = conn.execute(query, args)
    if commit:
        conn.commit()
        cursor.close()
        conn.close()
    else:
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        if results:
            if one:
                return results[0]
            return results
    return []


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def row_factory(cursor, data):
    keys = [col[0] for col in cursor.description]
    d = {k: v for k, v in zip(keys, data)}
    return dotdict(d)


def get_db_conn():
    conn = sqlite3.connect(CONSTS.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = row_factory
    return conn


@contextmanager
def get_cursor():
    try:
        conn = sqlite3.connect(CONSTS.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = row_factory
        cursor = conn.cursor()
        yield cursor
    finally:
        cursor.close()
        conn.close()
