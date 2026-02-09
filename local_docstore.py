# local_docstore.py
import os
import dbm
from typing import Optional, Iterator

class LocalDocStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def set(self, key: str, value: bytes):
        with dbm.open(self.path, 'c') as db:
            db[key] = value

    def get(self, key: str) -> Optional[bytes]:
        with dbm.open(self.path, 'r') as db:
            return db.get(key)

    def yield_keys(self) -> Iterator[str]:
        if not os.path.exists(self.path + ".bak") and not os.path.exists(self.path + ".dat") and not os.path.exists(self.path):
             return
        with dbm.open(self.path, 'r') as db:
            for key in db.keys():
                yield key.decode('utf-8') if isinstance(key, bytes) else key