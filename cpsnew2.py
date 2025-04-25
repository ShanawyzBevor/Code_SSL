import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
from model import VNet
from dataset2 import LAHeart, RandomCrop, RandomNoise, RandomRotFlip, ToTensor

# Dice Loss (Only for Supervised)
def dice_loss(score, target):
    target = target.long()
    score = F.softmax(score, dim=1)
    smooth = 1e-5
    target_onehot = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
    intersect = torch.sum(score * target_onehot)
    y_sum = torch.sum(target_onehot * target_onehot)
    z_sum = torch.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - dice  # loss

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_a = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)
model_b = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)

# DataLoader
batch_size = 4
train_transform = transforms.Compose([
    RandomRotFlip(),
    RandomNoise(),
    RandomCrop((112, 112, 80)),
    ToTensor(),
])
trainset = LAHeart(split='Training Set', label=True, transform=train_transform)
unlabelled_trainset = LAHeart(split='Training Set', label=False, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
unlabelled_trainloader = torch.utils.data.DataLoader(unlabelled_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# Optimizer & Scheduler
Max_epoch = 800
learn_rate = 0.0005
optimizer_a = optim.AdamW(model_a.parameters(), lr=learn_rate, weight_decay=1e-4)
optimizer_b = optim.AdamW(model_b.parameters(), lr=learn_rate, weight_decay=1e-4)
scheduler_a = optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=Max_epoch)
scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=Max_epoch)

# TensorBoard
writer = SummaryWriter()

# Training Loop
for epoch in range(Max_epoch):
    print(f'Epoch {epoch+1}/{Max_epoch}')
    print('-' * 30)
    model_a.train()
    model_b.train()

    total_sup_loss = 0.0
    total_dice = 0.0

    for batch_idx, sample in tqdm(enumerate(trainloader), total=len(trainloader)):
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        outputs_a = model_a(images)
        outputs_b = model_b(images)

        seg_loss_a = F.cross_entropy(outputs_a, labels)
        seg_loss_b = F.cross_entropy(outputs_b, labels)
        dice_loss_a = dice_loss(outputs_a, labels)

        sup_loss = seg_loss_a + seg_loss_b + dice_loss_a
        sup_loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        total_sup_loss += sup_loss.item()
        total_dice += (1 - dice_loss_a.item())

    avg_sup_loss = total_sup_loss / len(trainloader)
    avg_dice = total_dice / len(trainloader)

    print(f"Epoch {epoch+1}, Loss: {avg_sup_loss:.4f}, Dice: {avg_dice:.4f}")
    writer.add_scalar("Supervised Loss", avg_sup_loss, epoch)
    writer.add_scalar("Dice Accuracy", avg_dice, epoch)

    scheduler_a.step()
    scheduler_b.step()

    # Semi-Supervised with CPS after 100 Epoch
    if epoch >= 100:
        total_unsup_loss = 0.0
        for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader), total=len(unlabelled_trainloader)):
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            images = sample["image"].to(device)
            outputs_a = model_a(images)
            outputs_b = model_b(images)

            _, hardlabel_a = torch.max(outputs_a, dim=1)
            _, hardlabel_b = torch.max(outputs_b, dim=1)

            cps_loss = 0.01 * (F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a))
            cps_loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            total_unsup_loss += cps_loss.item()

        avg_unsup_loss = total_unsup_loss / len(unlabelled_trainloader)
        writer.add_scalar("Unsupervised CPS Loss", avg_unsup_loss, epoch)

    # Save Model
    if (epoch + 1) % 20 == 0:
        print("Saving model checkpoint...")
        torch.save(model_a.state_dict(), f"model_a_epoch_{epoch+1}.pth")
        torch.save(model_b.state_dict(), f"model_b_epoch_{epoch+1}.pth")

writer.flush()
writer.close()
