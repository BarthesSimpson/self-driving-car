# **Traffic Sign Recognition** 

## Writeup

Please review Traffic_Sign_Classifier for my project code, including the Dataset Exploration and Test a Model on New Images
requirements. In this writeup, I will cover the other items in the rubrik, mostly under the Design and Test a Model Architecture section:

### The submission describes the preprocessing techniques used and why these techniques were chosen.

Initially, I tried just normalizing the images (to speed up learning) and doing no further processing. Weirdly, the model performed worse than expected (c. 90% accuracy). When I added grayscaling, the performance improved (c. 93% accuracy). I had expected the color information would help, but perhaps the LeNet architecture is not deep enough for color information to be useful, and instead it just adds noise. I also shuffled the data to avoid overfitting.

### The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

I just used a standard LeNet model (the same one from a previous homework). It has the following layers:

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Relu Activation.
    # Max Pooling. Input = 28x28x6. Output = 14x14x6.

    # Layer 2: Convolutional. Output = 10x10x16.
    # Relu Activation.
    # Max Pooling. Input = 10x10x16. Output = 5x5x16.
    # Flatten. Input = 5x5x16. Output = 400.
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    # Relu Activation.

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    # Relu Activation.

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    
### The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

I trained the model with an Adam optimizer and the following hyperparameters:

learning_rate=0.0009
EPOCHS=60
BATCH_SIZE=128

### The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

I tuned the hyperparameters a bit to get high enough accuracy. In particular, the choice of learning rate made a big difference, and I had to use a smaller learning rate than was originally suggested.

### The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

I tried to pick images that the model would be successful with: clear, closely cropped, and at direct angles (similar to the images in the training data). I was fairly successful in this regard as the model got 4 out of the 5 images correct. The one it failed on was a "no vehicles" sign, which is difficult because it is (1) underrepresented in the training data, and (2) similar in appearance to the various speed limit signs. Indeed, the model gave it 11% probability of being a 50k/h sign, and 10% probability of being a no vehicles sign, so with a little more data (e.g. by augmenting the training dataset to add more no vehicles samples) the model would no doubt have classified it correctly.

