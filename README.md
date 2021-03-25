# Fully Convolutional Network with PyTorch
A toy experiment of FCN given VOC2012 Datasets.
## Usage
```bash
git clone https://github.com/JamesHsu333/Fully_Convolutional_Network.git
cd FCN
pip install -r requirements.txt
```
## Dataset
1. Download from 
[VOC2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
2. Extract directory ``` JPEGImages``` and ``` SegmentationClass``` under directory ```data```
## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
              VGG-32  [[-1, 64, 112, 112], [-1, 128, 56, 56], [-1, 256, 28, 28], [-1, 512, 14, 14], [-1, 512, 7, 7]]               0
  ConvTranspose2d-33          [-1, 512, 14, 14]       1,049,088
      BatchNorm2d-34          [-1, 512, 14, 14]           1,024
             ReLU-35          [-1, 512, 14, 14]               0
       UpSampling-36          [-1, 512, 14, 14]               0
  ConvTranspose2d-37          [-1, 256, 28, 28]         524,544
      BatchNorm2d-38          [-1, 256, 28, 28]             512
             ReLU-39          [-1, 256, 28, 28]               0
       UpSampling-40          [-1, 256, 28, 28]               0
  ConvTranspose2d-41          [-1, 128, 56, 56]         131,200
      BatchNorm2d-42          [-1, 128, 56, 56]             256
             ReLU-43          [-1, 128, 56, 56]               0
       UpSampling-44          [-1, 128, 56, 56]               0
  ConvTranspose2d-45         [-1, 64, 112, 112]          32,832
      BatchNorm2d-46         [-1, 64, 112, 112]             128
             ReLU-47         [-1, 64, 112, 112]               0
       UpSampling-48         [-1, 64, 112, 112]               0
  ConvTranspose2d-49         [-1, 32, 224, 224]           8,224
      BatchNorm2d-50         [-1, 32, 224, 224]              64
           Conv2d-51         [-1, 21, 224, 224]             693
================================================================
Total params: 16,463,253
Trainable params: 16,463,253
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 296.87
Params size (MB): 62.80
Estimated Total Size (MB): 360.25
----------------------------------------------------------------
```
## Quickstart
1.  Created a ```base_model``` directory under the ```experiments``` directory. It contains a file ```params.json``` which sets the hyperparameters for the experiment. It looks like
```Json
{
    "learning_rate": 0.0001,
    "batch_size": 10,
    "num_epochs": 10,
    "dropout_rate": 0.0,
    "num_channels": 32,
    "save_summary_steps": 100,
    "num_workers": 4
}
```
2. Train your experiment. Run
```bash
python train.py
```
3. Created a new directory ```learning_rate``` in experiments. Run
```bash
python search_hyperparams.py --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in ```search_hyperparams.py``` and create a new directory for each experiment under ```experiments/learning_rate/```.
4. Display the results of the hyperparameters search in a nice format
```bash
python synthesize_results.py --parent_dir experiments/learning_rate
```
5. Evaluation on the test set Once you've run many experiments and selected your best model and hyperparameters based on the performance on the validation set, you can finally evaluate the performance of your model on the test set. Run
```bash
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
## Resources
* For more Project Structure details, please refer to [Deep Learning Project Structure](https://deeps.site/blog/2019/12/07/dl-project-structure/)