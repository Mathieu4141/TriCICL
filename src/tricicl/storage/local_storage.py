from pathlib import PurePath
from typing import List

from tricicl.storage.remote_storage import RemoteStorageABC, SyncPath


class LocalStorage(RemoteStorageABC):
    """This storage does nothing"""

    def upload_file(self, sync_path: SyncPath):
        pass

    def download_file(self, sync_path: SyncPath):
        pass

    def list_files(self, remote_path: PurePath, suffix: str = "") -> List[PurePath]:
        return []
