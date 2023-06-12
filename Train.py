import torch
import torch.nn as nn
import torchvision
from Models.Discriminator import Discriminator
from Models.Generator import Generator
import torch.optim as optim
import Config
from torchvision.utils import save_image
from utils import save_checkpoint, load_checkpoint, save_some_examples
from tqdm import tqdm
from dataset import MapDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
NUM_EPOCHS = 10
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10

print(DEVICE)


def train(disc, gen, loader, opti_disc, opti_gen, l1_loss, criterion, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)
    for batch_idx, (x,y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            disc_real = disc(x,y)
            disc_lossreal = criterion(disc_real,torch.ones_like(disc_real))
            disc_fake = disc(x,y_fake)
            disc_lossfake = criterion(disc_fake,torch.zeros_like(disc_fake))
            disc_loss = (disc_lossreal+disc_lossfake)/2
        
        disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opti_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            gen_fake = gen(x)
            disc_fake = disc(x,gen_fake)
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            gen_loss = criterion(disc_fake, torch.ones_like(disc_fake))

        gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opti_gen)
        g_scaler.update()
        
        
def main():

    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3,features=64).to(DEVICE)
    opti_disc = optim.Adam(disc.parameters(),lr=LR,betas=(0.5,0.999))
    opti_gen = optim.Adam(gen.parameters(),lr=LR,betas=(0.5,0.999))
    criterion = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if Config.LOAD_MODEL:
        load_checkpoint(
            Config.CHECKPOINT_GEN, gen, opti_gen, LR,
        )
        load_checkpoint(
            Config.CHECKPOINT_DISC, disc, opti_disc,LR,
        )

    train_dataset = MapDataset(root_dir=Config.TRAIN_DIR,)
    train_dataloader = DataLoader(
                            train_dataset,
                            shuffle=True,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS
                            )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    val_dataset = MapDataset(root_dir = Config.VAL_DIR)
    val_dataloader = DataLoader(
                            val_dataset,
                            shuffle=True,
                            batch_size=1
                            )
    
    for epoch in range(NUM_EPOCHS):
        train(disc, gen, train_dataloader, opti_disc, opti_gen, L1_LOSS, criterion, g_scaler, d_scaler)

    if Config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opti_gen, filename=Config.CHECKPOINT_GEN)
            save_checkpoint(disc, opti_disc, filename=Config.CHECKPOINT_DISC)

    save_some_examples(gen, val_dataloader, epoch, folder="evaluation")
    
    if __name__ == "__main__":
        main()
