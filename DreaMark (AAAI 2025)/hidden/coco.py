
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader


import hidden.utils_img as utils_img

class COCODataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_list = glob(f"{root}/*.jpg")
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = default_loader(img_path)
        return self.transform(img)


class COCODataLoader(DataLoader):
    def __init__(self, batch_size, split, img_size, shuffle=True, normalize=True, **dataset_kwargs):
        if split == "train":
            ops = [
                transforms.RandomResizedCrop(img_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            if normalize:
                ops.append(utils_img.normalize_rgb)

            transform = transforms.Compose(ops)
        elif split == "val":
            ops = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
            if normalize:
                ops.append(utils_img.normalize_rgb)
            
            transform = transforms.Compose(ops)
        else:
            raise ValueError("split must be train or val")
        
        dataset = COCODataset(transform=transform, **dataset_kwargs)

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)