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

# Improved Dice Loss
def dice_loss(score, target, acc=0):
    target = target.float()
    score = F.softmax(score, dim=1)
    smooth = 1e-5
    intersect = torch.sum(score[:, 1, ...] * target[:, 1, ...])
    y_sum = torch.sum(target[:, 1, ...] * target[:, 1, ...])
    z_sum = torch.sum(score[:, 1, ...] * score[:, 1, ...])
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    if acc == 1:
        return dice
    return 1 - dice

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_a = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)
model_b = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)

# DataLoader
batch_size = 2
num_workers = 4
train_transform = transforms.Compose([
    RandomRotFlip(),
    RandomNoise(),
    RandomCrop((112, 112, 80)),
    ToTensor(),
])

trainset = LAHeart(split='Training Set', label=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)

unlabelled_trainset = LAHeart(split='Training Set', label=False, transform=train_transform)
unlabelled_trainloader = torch.utils.data.DataLoader(unlabelled_trainset, batch_size=batch_size,
                                                     shuffle=True, pin_memory=True,
                                                     num_workers=num_workers)

# Optimizer & Scheduler
Max_epoch = 800
learn_rate = 0.001
optimizer_a = optim.SGD(model_a.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)
optimizer_b = optim.SGD(model_b.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)

lr_scheduler_a = optim.lr_scheduler.StepLR(optimizer_a, step_size=1, gamma=np.power(0.1, 1 / Max_epoch))
lr_scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step_size=1, gamma=np.power(0.1, 1 / Max_epoch))

# TensorBoard
writer = SummaryWriter()

# Create directories for saving images
os.makedirs("./image", exist_ok=True)
os.makedirs("./pred", exist_ok=True)
os.makedirs("./label", exist_ok=True)

# Training Loop
for epoch in range(Max_epoch):
    print(f'Epoch {epoch+1}/{Max_epoch}')
    print('-' * 30)
    model_a.train()
    model_b.train()

    sup_loss_cps = 0.0
    sup_loss_seg_a = 0.0
    sup_loss = 0.0
    dice_acc_a = 0.0

    for batch_idx, sample in tqdm(enumerate(trainloader)):
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        outputs_a = model_a(images)
        outputs_b = model_b(images)

        _, hardlabel_a = torch.max(outputs_a, dim=1)
        _, hardlabel_b = torch.max(outputs_b, dim=1)

        cps_loss = F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a)
        seg_loss_a = F.cross_entropy(outputs_a, labels)
        seg_loss_b = F.cross_entropy(outputs_b, labels)

        dice_a = dice_loss(outputs_a, labels, 1)
        dice_acc_a += dice_a

        loss = seg_loss_a + seg_loss_b + 0.1 * cps_loss
        loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        sup_loss_cps += cps_loss.item()
        sup_loss_seg_a += seg_loss_a.item()
        sup_loss += loss.item()

    sup_loss_cps /= len(trainset)
    sup_loss_seg_a /= len(trainset)
    dice_acc_a /= len(trainset)
    sup_loss /= len(trainset)

    print(f"Labelled Dice: {dice_acc_a:.4f}")
    print(f"Labelled Loss: {sup_loss:.4f}")

    writer.add_scalar("CPS Loss/Labelled", sup_loss_cps, epoch)
    writer.add_scalar("SegLoss Model A/Labelled", sup_loss_seg_a, epoch)
    writer.add_scalar("Dice Accuracy/ Labelled", dice_acc_a, epoch)

    unsup_loss_cps = 0.0
    dice_acc_a = 0.0

    if epoch > 10:
        for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader)):
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            images = sample["image"].to(device)
            labels = sample["label"].to(device)

            outputs_a = model_a(images)
            outputs_b = model_b(images)

            _, hardlabel_a = torch.max(outputs_a, dim=1)
            _, hardlabel_b = torch.max(outputs_b, dim=1)

            unsup_cps_loss = 0.01 * (F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a))
            unsup_cps_loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            dice_a = dice_loss(outputs_a, labels, 1)
            dice_acc_a += dice_a
            unsup_loss_cps += unsup_cps_loss.item()

        unsup_loss_cps /= len(unlabelled_trainset)
        dice_acc_a /= len(unlabelled_trainset)
        print(f"Unlabelled Dice: {dice_acc_a:.4f}")

        writer.add_scalar("CPS Loss/Unlabelled", unsup_loss_cps, epoch)
        writer.add_scalar("Dice Accuracy/Unlabelled", dice_acc_a, epoch)

    if (epoch + 1) % 20 == 0:
        print("Saving model checkpoint...")
        torch.save(model_a.state_dict(), f"model_a_epoch_{epoch+1}.pth")
        torch.save(model_b.state_dict(), f"model_b_epoch_{epoch+1}.pth")

    if (epoch + 1) % 40 == 0:
        image3d = images.detach().cpu().numpy()
        label3d = labels.detach().cpu().numpy()
        pred3d = hardlabel_a.detach().cpu().numpy()

        for i in range(3):
            imageslice = image3d[0][0][:, :, i*20]
            labelslice = label3d[0][:, :, i*20]
            predslice = pred3d[0][:, :, i*20]
            for index in range(112):
                for j in range(112):
                    if imageslice[index][j] != 0:
                        imageslice[index][j] *= 15
                    if labelslice[index][j] != 0:
                        labelslice[index][j] = 255
                    if predslice[index][j] != 0:
                        predslice[index][j] = 255

            im = Image.fromarray(np.int8(imageslice)).convert('L')
            impath = f"./image/{epoch+1}_{i}.png"
            im.save(impath)

            pr = Image.fromarray(np.int8(predslice)).convert('L')
            prpath = f"./pred/{epoch+1}_{i}.png"
            pr.save(prpath)

            la = Image.fromarray(np.int8(labelslice)).convert('L')
            lapath = f"./label/{epoch+1}_{i}.png"
            la.save(lapath)

    lr_scheduler_a.step()
    lr_scheduler_b.step()

writer.flush()
writer.close()
