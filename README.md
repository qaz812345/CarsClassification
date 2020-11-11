# Cars Classification
This is a car image classification task for custom [Stanford Cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). We fine-tune the pre-trained ResNeXt-101-32x8d which provided in [torchvision models](https://pytorch.org/docs/stable/torchvision/models.html). The highest testing accuracy can reach 91.64%.

#### In train.py, it will do:
1. Define CarsData class for custom dataset
2. Create model and set hyperparameters
3. Load data and pre-process
4. Train model
5. Save model
6. Plot training loss curve and accuracy curve
7. Inference

#### Model Compar

   |Model | Testing Accuracy (%) |
   |:------: | :-----------: |
   |Wide ResNet-50-2 | 91.04 |
   |ResNeXt-101-32x8d | 91.64 |

## Reproducing Submission
1. [Installation](#Installation)
2. [Dataset](#Dataset)
3. [Training](#Training)
4. [Inference](#Inference)
## Installation
Use pip to install python packages from requirements.txt.

```pip install -r requirements.txt```

## Dataset
We use custom Stanford Cars dataset as our data. Click [here](https://www.kaggle.com/c/cs-t0828-2020-hw1/data) to download the dataset.

There are 196 car classes in the dataset. We have 11,185 images for training and 5,000 for testing. We divide the training data into 10,000 images for training and 1,185 images for validation.

To load data without modify the code, you need to set the data directory structure as:
```
data
+- training_labels.csv
+- training_data
|  +- training_data
|     +- training_image.jpg
|     +- ...
+- testing_data
|  +- testing_data
|     +- testing_image.jpg
|     +- ...
```
or you can pass the path of data directory and the label csv file while constructing the CarsDataset object.

## Training
### Data Pre-process and Augmentation
* Resize image
*	Random crop image
*	Random horizontal flip
*	Color jitter
*	Random rotation
*	Image normalization
### Model
* Pre-trained ResNeXt-101-32x8d
### Hyperparameters
*	Epochs: 100
*	Batch size: 32
*	Optimizer: SGD (learn rate=0.001, momentum=0.9)

At the end of training loop, we will save the model parameter file to models directory.

## Inference
Load the testing images and start inference, and you will get an output csv file at the end. 

