'''
Reference: https://github.com/AIPI540/AIPI540-Deep-Learning-Applications/blob/main/2_computer_vision/CNNs/transfer_learning.ipynb
Author: Jon Reifschneider
Original repository link: https://github.com/AIPI540/AIPI540-Deep-Learning-Applications
'''

# Import necessary modules
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Check GPU devices
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Hyperparameters
LR = 0.001
BATCH_SIZE = 10
GAMMA = 0.2
NUM_EPOCH = 10
class_names = ['artificial', 'human']

# Pre-processing 
def load_and_transform_dataset(data_dir):
    '''
    Load the datasets from the data directory path and transform them to PyTorch
    datasets and dataloaders for training, validation and tests

    Input:
        data_dir: String
    
    Output:
        dataloaders: dict of String: torch.utils.data.DataLoader
        dataset_sizes: dict of String: int
    '''
    # Define the transforms to be applied to the images
    transform_train = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=30),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)), # normalize the pixel values according to the mean and std of ImageNet
    ])

    transform_val_test = T.Compose([
        T.Resize(256), 
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)), # normalize the pixel values according to the mean and std of ImageNet
    ])

    # Apply the training transform to the training set and load the training set 
    dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Apply the validation & test transform to the validation set and load the validation set 
    dataset_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_val_test)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Apply the validation & test transform to the test set and load the test set 
    dataset_test = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform_val_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True, num_workers=4)

    dataloaders = {'train': dataloader_train, 
                'val': dataloader_val, 
                'test': dataloader_test}
    dataset_sizes = {'train': len(dataset_train), 
                    'val': len(dataset_val), 
                    'test': len(dataset_test)}
    return dataloaders, dataset_sizes

data_dir = 'data/final_output_data'
dataloaders, dataset_sizes = load_and_transform_dataset(data_dir)

def get_model():
    '''
    Define model architecture: deep CNN model applied transfer learning based on 
    pretrained resnet-18 model

    Input: 
        None

    Output:
        model: torchvision.models.resnet.ResNet
    '''
    # Instantiate pre-trained resnet
    model = torchvision.models.resnet18(pretrained=True)
    # Shut off autograd for all layers to freeze model so the layer weights are not trained
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the number of inputs to final Linear layer
    num_features = model.fc.in_features

    # Replace final Linear layer with a new Linear with the same number of inputs but just 2 outputs (2 classes)
    model.fc = nn.Linear(num_features, 2)

    return model

model = get_model()

# Define criterion
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.RMSprop(model.parameters(), lr=LR)

# Define learning rate schedule
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=GAMMA)

def train_model(model, criterion, optimizer, dataloaders, scheduler, device, num_epochs, defrost):
    '''
    Function for model training & evaluation

    Input:
        model: torchvision.models.resnet.ResNet
        criterion: torch.nn.modules.loss.CrossEntropyLoss
        optimizer: torch.optim.rmsprop.RMSprop
        dataloaders: dict of String: torch.utils.data.DataLoader
        scheduler: torch.optim.rmsprop.lr_scheduler.StepLR
        device: String
        num_epochs: int
        defrost: boolean
    
    Output: 
        model: torchvision.models.resnet.ResNet
    '''
    model = model.to(device) # Send model to GPU if available
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Get the input images and labels, and send to GPU if available
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * inputs.size(0)
                # Track number of correct predictions
                running_corrects += torch.sum(preds == labels.data)

            # Step along learning rate scheduler when in train
            if phase == 'train':
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # If model performs better on val set, save weights as the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:3f}'.format(best_acc))

    # Load the weights from best model
    model.load_state_dict(best_model_wts)

    # "defrost" all the model parameters by turning on autograd for all layers so the layer weights are also trained
    if defrost: 
        print("unfreeze the model and start training again:" )
        for param in model.parameters():
            param.requires_grad = True

        since = time.time()
        num_epochs_unfreeze = 5 # defrost parameter training with 5 epochs
        for epoch in range(1, num_epochs_unfreeze+1): 
            print('Epoch {}/{}'.format(epoch, num_epochs_unfreeze))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Get the input images and labels, and send to GPU if available
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the weight gradients
                    optimizer.zero_grad()

                    # Forward pass to get outputs and calculate loss
                    # Track gradient only for training data
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backpropagation to get the gradients with respect to each weight
                        # Only if in train
                        if phase == 'train':
                            loss.backward()
                            # Update the weights
                            optimizer.step()

                    # Convert loss into a scalar and add it to running_loss
                    running_loss += loss.item() * inputs.size(0)
                    # Track number of correct predictions
                    running_corrects += torch.sum(preds == labels.data)

                # Step along learning rate scheduler when in train
                if phase == 'train':
                    scheduler.step()

                # Calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # If model performs better on val set, save weights as the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:3f}'.format(best_acc))

        # Load the weights from best model
        model.load_state_dict(best_model_wts)

    return model

# Train and save the model
model = train_model(model, criterion, optimizer, dataloaders, lr_scheduler, device, num_epochs=NUM_EPOCH, defrost=True)
model_path = "models/model4.pth"
torch.save(model.state_dict(), model_path)

def test_model(model, model_path, dataloader_test):
    '''
    Function for model testing

    Input:
        model: torchvision.models.resnet.ResNet
        model_path: String
        dataloader_test: torch.utils.data.DataLoader
    
    Output: 
        y_true: numpy.ndarray
        y_pred: numpy.ndarray
    '''

    # load the existing model checkpoints
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) # Send model to GPU if available
    with torch.no_grad():
        model.eval()
        # Get all test images (every image in one batch)
        images, labels = next(iter(dataloader_test))
        images, labels = images.to(device), labels.to(device)
        # Get predictions
        _,preds = torch.max(model(images), 1)
        # convert predictions & labels to numpy array format
        y_pred = np.squeeze(preds.cpu().numpy())
        y_true = np.squeeze(labels.cpu().numpy())
    return y_true, y_pred

y_true, y_pred = test_model(model, model_path, dataloaders['test'])

# Classification report
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
disp.plot()
plt.show()