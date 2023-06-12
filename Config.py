import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

transform = A.Compose(
            [A.resize([256,256])],
            additional_targets={"image0": "image"}
)

input_transform = A.Compose([
    A.ColorJitter(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean = [0.5,0.5,0.5],syd = [0.5,0.5,0.5] , max_pixel_value=255.0),
    ToTensorV2()
])

mask_transform = A.Compose([
    A.Normalize(mean = [0.5,0.5,0.5],syd = [0.5,0.5,0.5] , max_pixel_value=255.0),
    ToTensorV2()
])