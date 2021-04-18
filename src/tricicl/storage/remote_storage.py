import shutil
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory
from typing import Optional


class RemoteStorageABC(ABC):
    def download_dataset(self, name: str):
        self.download_zipped_directory_if_missing(Path.home() / f".avalanche/data/{name}", PurePath(f"data/{name}"))

    def upload_dataset(self, name: str):
        self.upload_zipped_directory(Path.home() / f".avalanche/data/{name}", PurePath(f"data/{name}"))

    def upload_zipped_directory(self, local_path: Path, remote_path: PurePath):
        with TemporaryDirectory() as td_name:
            tar_path = (Path(td_name) / local_path.name).with_suffix(".tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(str(local_path), arcname="")
            return self.upload_file(tar_path, remote_path.with_suffix(".tar.gz"))

    def download_file_if_missing(self, local_path: Path, remote_path: Optional[PurePath] = None):
        if not local_path.exists():
            self.download_file(local_path, remote_path)

    def download_zipped_directory(self, local_path: Path, remote_path: PurePath):
        with TemporaryDirectory() as td_name:
            zip_temp = Path(td_name) / f"temp_{local_path.name}.tar.gz"
            self.download_file(zip_temp, remote_path.with_suffix(".tar.gz"))
            local_path.mkdir(exist_ok=True, parents=True)
            shutil.unpack_archive(zip_temp, local_path, "gztar")

    def download_zipped_directory_if_missing(self, local_path: Path, remote_path: PurePath):
        if local_path.exists():
            return
        self.download_zipped_directory(local_path, remote_path)

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: PurePath):
        pass

    @abstractmethod
    def download_file(self, local_path: Path, remote_path: PurePath):
        pass
