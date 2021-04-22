import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory
from typing import List

from tricicl.constants import PROJECT_DIR


@dataclass
class SyncPath:
    remote: PurePath
    local: Path

    @staticmethod
    def from_local(local_path: Path) -> "SyncPath":
        return SyncPath(remote=local_path.relative_to(PROJECT_DIR), local=local_path)

    @staticmethod
    def from_remote(remote_path: PurePath) -> "SyncPath":
        return SyncPath(remote=remote_path, local=PROJECT_DIR / remote_path)

    @staticmethod
    def from_str(rel_path: str) -> "SyncPath":
        return SyncPath.from_local(PROJECT_DIR / rel_path)


class RemoteStorageABC(ABC):
    def download_dataset(self, name: str):
        self.download_zipped_directory_if_missing(get_dataset_sync_path(name))

    def upload_dataset(self, name: str):
        self.upload_zipped_directory(get_dataset_sync_path(name))

    def upload_zipped_directory(self, sync_path: SyncPath):
        with TemporaryDirectory() as td_name:
            tar_path = shutil.make_archive(
                Path(td_name) / sync_path.local.relative_to(sync_path.local.anchor),
                "gztar",
                sync_path.local,
            )
            return self.upload_file(SyncPath(local=Path(tar_path), remote=sync_path.remote.with_suffix(".tar.gz")))

    def download_file_if_missing(self, sync_path: SyncPath):
        if not sync_path.local.exists():
            self.download_file(sync_path)

    def download_zipped_directory(self, sync_path: SyncPath):
        with TemporaryDirectory() as td_name:
            zip_temp = Path(td_name) / f"temp_{sync_path.local.name}.tar.gz"
            self.download_file(SyncPath(local=zip_temp, remote=sync_path.remote.with_suffix(".tar.gz")))
            sync_path.local.mkdir(exist_ok=True, parents=True)
            shutil.unpack_archive(zip_temp, sync_path.local, "gztar")

    def download_zipped_directory_if_missing(self, sync_path: SyncPath):
        if sync_path.local.exists():
            return
        self.download_zipped_directory(sync_path)

    def download_directory(self, sync_path: SyncPath):
        for remote_file in self.list_files(sync_path.remote):
            self.download_file_if_missing(SyncPath.from_remote(remote_file))

    def upload_directory(self, sync_path: SyncPath):
        for local_file in sync_path.local.glob("*"):
            if local_file.is_file():
                self.upload_file(SyncPath.from_local(local_file))

    @abstractmethod
    def upload_file(self, sync_path: SyncPath):
        pass

    @abstractmethod
    def download_file(self, sync_path: SyncPath):
        pass

    @abstractmethod
    def list_files(self, remote_path: PurePath, suffix: str = "") -> List[PurePath]:
        pass


def get_dataset_sync_path(name: str) -> SyncPath:
    return SyncPath(local=Path.home() / f".avalanche/data/{name}", remote=PurePath(f"data/{name}"))
