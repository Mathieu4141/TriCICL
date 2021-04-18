from pathlib import Path, PurePath

from google.cloud.storage import Bucket, Client
from memoized_property import memoized_property

from tricicl.storage.remote_storage import RemoteStorageABC


class GCStorage(RemoteStorageABC):
    def __init__(self, bucket_name: str = "tricicl-public"):
        self.bucket_name = bucket_name

    def upload_file(self, local_path: Path, remote_path: PurePath):
        blob = self.bucket.blob(str(remote_path), chunk_size=10 * 1024 * 1024)
        blob.upload_from_filename(str(local_path), timeout=60 * 5)

    def download_file(self, local_path: Path, remote_path: PurePath):
        local_path.parent.mkdir(exist_ok=True, parents=True)
        blob = self.bucket.get_blob(str(remote_path))
        if blob is None:
            raise FileNotFoundError(f"{remote_path} is not on gcloud bucket")
        blob.download_to_filename(str(local_path), timeout=60 * 5)

    @memoized_property
    def client(self) -> Client:
        return Client()

    @memoized_property
    def bucket(self) -> Bucket:
        return self.client.bucket(self.bucket_name)
