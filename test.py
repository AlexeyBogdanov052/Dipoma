#=============Тесты===============
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from DataSetTXT import Dataset
from Models.MobileNet_v2 import MobileNetV2

class ROC:

    tp = 0
    fp = 0
    tn = 0
    fn = 0

if __name__ == '__main__':

    dict_roc = dict()
    for thr in np.arange(0.0, 1.0, 0.001):
        dict_roc[thr] = ROC()
    tpr_list = []
    fpr_list = []

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU ...')
    else:
        print('CUDA is available! Training on GPU ...')

    batch_size = 1

    test_transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        ])

    test_data = Dataset(csv_file='C:/Diploma/CSV/Test.csv', root_dir='', transform=test_transform)

    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    model = MobileNetV2(n_class=2)
    model.load_state_dict(torch.load('model_cifar.pt'))
    #criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in test_dataloader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        print(output)
        for thr in np.arange(0.0, 1.0, 0.001):
            if (target == 1) and (output[0][1] > thr):
                dict_roc[thr].tp += 1
            if (target == 0) and (output[0][1] > thr):
                dict_roc[thr].fp += 1
            if (target == 0) and (output[0][1] < thr):
                dict_roc[thr].tn += 1
            if (target == 1) and (output[0][1] < thr):
                dict_roc[thr].fn += 1

    for thr in np.arange(0.0, 1.0, 0.001):
        tpr = dict_roc[thr].tp/(dict_roc[thr].tp + dict_roc[thr].fn)
        tpr_list.append(tpr)
        fpr = dict_roc[thr].fp/(dict_roc[thr].fp + dict_roc[thr].tn)
        fpr_list.append(fpr)
        print(thr, tpr, fpr)

    #=============================================================================
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_list, tpr_list, color = 'b', label = 'ROC')
    plt.semilogx([0,1], [0.7,1], color="navy", linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.show()
