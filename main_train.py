import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import albumentations as A

from DataSetTXT import Dataset
from Models.MobileNet_v2 import MobileNetV2

if __name__ == '__main__':

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU ...')
    else:
        print('CUDA is available! Training on GPU ...')

    num_workers = 0
    batch_size = 20
    WD = 0.1

    # train_transform = transforms.Compose([
    #    transforms.RandomCrop((224, 224)),
    #   transforms.RandomHorizontalFlip(p=0.5),
    #    ])

    train_transform = A.Compose([
        A.RandomCrop(224, 224),
        A.Flip(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

    val_transform = A.Compose([
        A.CenterCrop(224, 224)
    ])

    train_data = Dataset(csv_file='C:/Diploma/CSV/Train(1).csv', root_dir='', transform=train_transform)
    val_data = Dataset(csv_file='C:/Diploma/CSV/Validation(1).csv', root_dir='', transform=val_transform)


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)  # Будет тормозить, поставить поменьше
    val_dataloader = DataLoader(val_data, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    #model = MobileNetV2(n_class=2)
    #model.load_state_dict(torch.load('model_cifar.pt'))
    # print(model)
    model = MobileNetV2(n_class=1000)
    model.load_state_dict(torch.load('mobilenetv2_1.0-f2a8633.pth.tar'))
    model.classifier = nn.Linear(model.last_channel, 2)

    # =============================================================================#
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=WD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    n_epochs = 1
    valid_loss_min = np.inf

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_dataloader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in val_dataloader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)
        valid_loss = valid_loss / len(val_dataloader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'mobilenetv2_1.0-f2a8633.pth.tar')
            valid_loss_min = valid_loss

        scheduler.step()
