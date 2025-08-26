import cv2 as cv
from torch.utils.data import Dataset
import glob


class ScrewsDataset(Dataset):
    def __init__(self):
        self.strings = glob.glob('./Dataset/*.png')

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        return cv.imread(self.strings[idx]) 