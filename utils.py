import hashlib
import subprocess
from datetime import datetime
from functools import cache


def get_current_datetime_w_us_str():
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S_%f')


def get_dt_format():
    return "%Y-%m-%d %H:%M:%S"


def get_current_datetime():
    now = datetime.now()
    return datetime.strptime(now.strftime(get_dt_format()), get_dt_format())


@cache
def count_image_files(image_dir):
    result = subprocess.run(
        ['find', image_dir, '-type', 'f'],
        capture_output=True,
        text=True,
        check=True
    )
    file_count = len(result.stdout.splitlines())
    return file_count


def get_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()