from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epoch_loss_list = []
    epoch_loss_val_list = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
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
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {(best_acc * 100):.2f}%')
    print(f'Best best_epoch: {best_epoch}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch

if __name__ == '__main__':
    adam = False
    num_epochs = 20
    learning_rate = 0.0001
    SGD_momentum = 0.9
    step_size = 7
    gamma = 0.1
    batch_size = 6
    num_workers = 2
    shuffle = True
    seed = 8

    seed_everything(seed)
    cudnn.benchmark = True

    data_dir = 'C:\\repos\\VinchikGPT\\Forms\\402965562\\train_dir'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle = shuffle, num_workers = num_workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

    #model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))


    model_ft = model_ft.to(device)

    checkpoint = torch.load("classifier_68.35.pth", map_location = torch.device('cpu'))
    model_ft.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if adam == True:
        optimizer_ft = optim.Adam(model_ft.parameters(), lr = learning_rate)
    else:
        optimizer_ft = optim.SGD(model_ft.parameters(), lr = learning_rate, momentum = SGD_momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = step_size, gamma = gamma)

    model_ft, best_epoch = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs = num_epochs)
    


    '''
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 4)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    if adam == True:
        optimizer_ft = optim.Adam(model_conv.fc.parameters(), lr=learning_rate)
    else:
        optimizer_ft = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=SGD_momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model_ft, best_epoch = train_model(model_conv, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    '''


    state = {
        "epoch": best_epoch,
        "state_dict": model_ft.state_dict(),
        "optimizer": optimizer_ft.state_dict(),
    }
    torch.save(state, 'classifier.pth')