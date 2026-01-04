from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader


def create_celeba_dataset(data_path, batch_size, **kwargs):
    split = kwargs.get("split", "train")  # train / valid / test
    download = kwargs.get("download", True)
    image_size = kwargs.get("image_size", 64)

    dataset = CelebA(
        root=data_path,
        split=split,
        download=download,
        transform=transforms.Compose([
            # CelebA 官方常用裁剪方式（对齐人脸）
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])
    )

    loader_params = dict(
        shuffle=kwargs.get("shuffle", True),
        drop_last=kwargs.get("drop_last", True),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 4),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        **loader_params
    )

    return dataloader
