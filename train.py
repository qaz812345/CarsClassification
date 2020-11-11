import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils, models
from PIL import Image
from tqdm import tqdm

class CarsDataset(Dataset):
    """ A class for custom cars dataset.
    
    Attributes
    ----------
    csv_file : str
        a file path of csv file which has stored the correspondence between
        image id and label
    root_dir : str
        a directory path which specifies the location of image data
    data_range :tuple
        a range for accessing partial data (defult (0, ))
    transform : transform function
        a transform function for preprocessing PIL images
    classes : list
        a list of all class name in dataset
    class_dict : dict
        a dictionary with the correspondence between class name and label

    Methods
    -------
    get_classes()
        get all class names correspond to its label
    """

    def __init__(self, csv_file, root_dir, data_range=(0,), transform):
        """
        Parameters
        ----------
        csv_file : str
            a file path of csv file which has stored the correspondence between
            image id and label
        root_dir : str
            a directory path which specifies the location of image data
        data_range :tuple
            a range for accessing partial data (defult (0, ))
        transform : transform function
            a transform function for preprocessing PIL images
        """

        self.csv_file = pd.read_csv(csv_file, encoding='utf8')
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self.csv_file.sort_values(by='id')['label'].unique().tolist()
        self.class_dict = {k:i for i, k in enumerate(self.classes)}
        self.csv_file = self.csv_file[data_range[0] : data_range[1]]
        
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int
            an index for accessing element in dataset
        """

        image_path = f'{self.root_dir}/{self.csv_file.iloc[index, 0]:06d}.jpg'
        image = Image.open(image_path).convert('RGB')
        label_name = self.csv_file.iloc[index, 1]
        label = self.class_dict[label_name]
        image = self.transform(image)
        
        return image, label
    
    def get_classes(self):
        return self.classes


if __name__ == '__main__':
    # Dtetermine use CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # Set hyper-prarmeter
    batch_size = 32
    epochs = 100
    lr_rate = 0.001
       
    # Fine-tune pytorch pretrained model
    # net = models.wide_resnet50_2(pretrained=True)
    net = models.resnext101_32x8d(pretrained=True)
    # Replace the last fc layer
    num_ftrs = net.fc.in_features
    net.fc =nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 196)
    )
    net = net.to(device)
    print(net)
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=lr_rate)
    optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=0.9)
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Load data
    train_set = CarsDataset(csv_file='data/training_labels.csv',
                            root_dir='data/training_data/training_data/',
                            data_range=(0,10000),
                            transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_set = CarsDataset(csv_file='data/training_labels.csv',
                            root_dir='data/training_data/training_data/',
                            data_range=(10000,11185),
                            transform=test_transform)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = train_set.get_classes()
    
    # Statistics for visualization
    train_losses = []
    train_accuracies = []
    test_errors = []
    test_accuracies = []
    
    # Traing
    start = time.time()
    for epoch in range(epochs):
        net.train()
        total = 0.0
        running_loss = 0.0
        running_correct = 0.0
        for data in tqdm(train_loader):
            # Get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            total += len(labels)
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * running_correct/total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Training [epoch: {epoch+1}] loss: {epoch_loss:.4f} accuracy: {epoch_acc:.2f}')

        # Validation
        with torch.no_grad():
            net.eval()
            correct = 0.0
            total = 0.0
            for data in valid_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += len(labels)

            test_err = 100. * (total-correct)/total
            test_acc = 100. * correct/total
            test_errors.append(test_err)
            test_accuracies.append(test_acc)
            print(f'Validation [epoch: {epoch+1}] accuracy： {test_acc:.2f}%')

    duration = time.time()-start
    print(f'Finished Training! Duration: {duration}')

    # Save model
    PATH = 'models/resnext101.pth'
    torch.save(net.state_dict(), PATH)

    # Plot training loss curve
    times = [i for i in range(epochs)]
    plt.figure(figsize=(8,15), dpi=100, linewidth=2)
    plt.subplot(121),plt.plot(times, train_losses, '-', color='r', label="loss")
    plt.ylabel("loss", fontsize=20, labelpad=15)
    plt.title('Training Loss', fontsize=20), plt.xticks(), plt.yticks(), plt.legend()

    # Plot training and testing error rate curve
    plt.subplot(122)
    plt.plot(times, train_accuracies, '-', color='b', label="training")
    plt.plot(times, test_accuracies, '-', color='r', label="testing")
    plt.ylabel("accuracy", fontsize=20, labelpad=15)
    plt.title('Accuracy', fontsize=20), plt.xticks(), plt.yticks(), plt.legend()

    plt.savefig('results.png')

    # Testing
    test_root_dir = 'data/testing_data/testing_data'
    test_imgs = os.listdir(test_root_dir) 
    result = pd.DataFrame({'id':test_imgs, 'label':None})
    result['id'] = result['id'].apply(lambda x: x.split('.')[0])

    net.eval()
    for i in range(len(result)):
        img = Image.open(f'{test_root_dir}/{result.iloc[i,0]}.jpg').convert('RGB')
        img = test_transform(img).float()
        img = torch.autograd.Variable(img, requires_grad=True)
        img = img.unsqueeze(0)
        img = img.to(device)
        out = net(img)
        conf, predicted = torch.max(out.data, 1)
        result.iloc[i,1] = classes[predicted]

    result.to_csv('testing_results.csv', encoding='utf8', index=False)
