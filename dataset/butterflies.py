from datasets import load_dataset

# ds = load_dataset("huggan/smithsonian_butterflies_subset")
from torch.utils.data import Dataset
from PIL import Image


class SmithsonianButterfliesDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = load_dataset(
            "huggan/smithsonian_butterflies_subset",
            split=split
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]          # PIL.Image
        if self.transform is not None:
            img = self.transform(img)
        return img, 0                # 和 CIFAR10 对齐，dummy label




from torchvision import transforms
from torch.utils.data import DataLoader

def create_butterflies_dataset(data_path=None, batch_size=64, **kwargs):
    train = kwargs.get("train", True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = SmithsonianButterfliesDataset(
        split="train" if train else "test",
        transform=transform
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
