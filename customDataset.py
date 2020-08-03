import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
class PlantDataset(Dataset):
    def __init__(self,anno,root_dir=None,transform=None):
        self.annotations=anno
        self.transform=transform
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path=self.annotations[index][0]
        image=cv2.imread(img_path,cv2.IMREAD_COLOR)
        y_label=torch.tensor(int(self.annotations[index][1]))
        if self.transform:
          image=self.transform(image)
        
        return (image,y_label)
