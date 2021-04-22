from pathlib import PurePath
from typing import List

from google.cloud.storage import Client

from tricicl.storage.remote_storage import RemoteStorageABC, SyncPath


class GCStorage(RemoteStorageABC):
    def __init__(self, project: str = "mathieu-tricicl", bucket_name: str = "tricicl-public"):
        self.client = Client(project=project)
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, sync_path: SyncPath):
        blob = self.bucket.blob(str(sync_path.remote), chunk_size=10 * 1024 * 1024)
        blob.upload_from_filename(str(sync_path.local), timeout=60 * 5)

    def download_file(self, sync_path: SyncPath):
        sync_path.local.parent.mkdir(exist_ok=True, parents=True)
        blob = self.bucket.get_blob(str(sync_path.remote))
        if blob is None:
            raise FileNotFoundError(f"{sync_path.remote} is not on gcloud bucket")
        blob.download_to_filename(str(sync_path.local), timeout=60 * 5)

    def list_files(self, remote_path: PurePath, suffix: str = "") -> List[PurePath]:
        return [
            PurePath(b.name)
            for b in self.client.list_blobs(self.bucket, prefix=str(remote_path))
            if not b.name.endswith("/") and b.name.endswith(suffix)
        ]
