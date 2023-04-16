from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import time
import os
import copy
from pytorch_lightning import seed_everything

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(user_path, message_id, user_id, vk):
    num_epochs = 10
    learning_rate = 0.0001
    SGD_momentum = 0.9
    step_size = 7
    gamma = 0.1
    batch_size = 6
    num_workers = 2
    shuffle = True
    seed = 8
    print("\n")
    seed_everything(seed)
    cudnn.benchmark = True
    image_datasets = {x: datasets.ImageFolder(os.path.join(user_path + "\\train_dir", x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = shuffle, num_workers = num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))
    model = model.to(device)

    checkpoint = torch.load("girl_classifier_68.pth", map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = SGD_momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_list = []
    epoch_loss_val_list = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        vk.method('messages.edit', {'message_id': message_id, 'peer_id': user_id, 'user_id': user_id, 'message': f"Обучение: {epoch + 1}/{num_epochs}"})
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                epoch_loss_list.append(epoch_loss)
            if phase == 'val':
                epoch_loss_val_list.append(epoch_loss)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {(epoch_acc * 100):.2f}%')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {(best_acc * 100):.2f}%')
    print(f'Best best_epoch: {best_epoch}')
    model.load_state_dict(best_model_wts)
    state = {
        "epoch": best_epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, user_path + "\\classifier.pth")
    return best_acc