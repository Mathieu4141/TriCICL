from tricicl.storage.gcloud import GCStorage

if __name__ == "__main__":
    # GCStorage().upload_dataset("mnist")
    GCStorage().upload_dataset("cifar100")
