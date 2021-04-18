from pathlib import Path, PurePath

from google.cloud.storage import Client

from tricicl.storage.remote_storage import RemoteStorageABC


class GCStorage(RemoteStorageABC):
    def __init__(self, project: str = "mathieu-tricicl", bucket_name: str = "tricicl-public"):
        self.client = Client(project=project)
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path: Path, remote_path: PurePath):
        blob = self.bucket.blob(str(remote_path), chunk_size=10 * 1024 * 1024)
        blob.upload_from_filename(str(local_path), timeout=60 * 5)

    def download_file(self, local_path: Path, remote_path: PurePath):
        local_path.parent.mkdir(exist_ok=True, parents=True)
        blob = self.bucket.get_blob(str(remote_path))
        if blob is None:
            raise FileNotFoundError(f"{remote_path} is not on gcloud bucket")
        blob.download_to_filename(str(local_path), timeout=60 * 5)
