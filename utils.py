import hashlib
from datetime import datetime
from functools import cache
from pathlib import Path
from time import perf_counter

from consts import valid_extensions


def get_current_datetime_w_us_str() -> str:
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S_%f')


def get_dt_format() -> str:
    return "%Y-%m-%d %H:%M:%S"


def get_current_datetime() -> datetime:
    now = datetime.now()
    return datetime.strptime(now.strftime(get_dt_format()), get_dt_format())


@cache
def count_image_files(directory) -> int:
    print('Counting files...')
    file_count = sum(1 for p in Path(directory).rglob('*') if p.is_file() and p.suffix.lower() in valid_extensions)
    print('Finished.')
    return file_count


def get_sha256(file_path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sort_two_lists(list_leader, list_follower, desc=True):
    combined = list(zip(list_leader, list_follower))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=desc)
    sorted_leader, sorted_follower = zip(*sorted_combined) if sorted_combined else ([], [])
    return list(sorted_leader), list(sorted_follower)


class Perf:
    __slots__ = ('previous', 'checkpoints', 'topic')

    def __init__(self, topic: str=None):
        self.topic = topic
        self.checkpoints = []
        self.previous = perf_counter()

    def check(self, name: str=""):
        now = perf_counter()
        elapsed = now - self.previous
        self.previous = now
        self.checkpoints.append((name, elapsed))
        return elapsed

    def __repr__(self) -> str:
        total = sum(point[1] for point in self.checkpoints)
        longest = max(max(len(point[0]) for point in self.checkpoints), 5) # 5 is len of 'total'
        topic = f'[{self.topic}]\n' if self.topic else ''
        return topic + '\n'.join(
            f'{name:<{longest}}: {elapsed:.4f} {elapsed / total * 100 :.1f}%'
            for name, elapsed in self.checkpoints
        ) + f'\n{"total":<{longest}}: {total:.4f}'
