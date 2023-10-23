# Binary_Image_Classifier
 PyTorch implementation for a Convolutional Neural Network (CNN) classifier

**Objective:** Image classification using a deep learning model.
**Framework:** The code is written in PyTorch, a popular deep learning framework.

**1. Data Preparation:**

* Image dataset is located in the 'face' directory.
* Data is divided into training, validation, and test sets.
* Data preprocessing is performed, including resizing, data augmentation (e.g., flips, rotations), and normalization.

**2. Data Handling:**

* PyTorch's ImageFolder dataset is used for data loading.
* Data is converted into normalized PyTorch tensors.
* Data loaders are created for training, validation, and test datasets.

**3. Model Architecture:**

* A custom Convolutional Neural Network (CNN) model is defined using PyTorch's nn.Module.
* The model consists of convolutional layers, pooling layers, fully connected layers, and dropout layers.
* ReLU and LogSoftmax activation functions are used.

**4. Training:**

* The model is trained using the training dataset.
* Training involves forward and backward passes, loss calculation, and optimization.
* PyTorch's automatic differentiation (autograd) is used for gradient calculation.

**5. Validation:**

* Model performance is evaluated on the validation dataset during training.
* Validation loss is tracked, and the model parameters are saved if the loss decreases.

**6. Testing:**

* The trained model is tested on the test dataset.
* Test loss and accuracy are calculated.

**7. Results:**

* Test accuracy, class-wise accuracies, and overall accuracy are reported.
* PyTorch's torch.save is used to save the model's parameters.

**8. CUDA Usage:**

* The code checks for the availability of a CUDA-capable GPU and moves data and model to the GPU if available.

**9. Project Overview:**

* This project aims to classify images into two classes.
* While the code provides a basic structure, real-world projects often involve more complexity and fine-tuning for specific tasks.
