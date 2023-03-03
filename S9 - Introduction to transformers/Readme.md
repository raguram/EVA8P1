# Assignment 9 - Introduction to transformers 

## **Problem statement**

Build the following network:

* That takes a CIFAR10 image (32x32x3)
* Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
* Apply GAP and get 1x1x48, call this X
* Create a block called ULTIMUS that:
    * Creates 3 FC layers called K, Q and V such that:
        * X*K = 48*48x8 > 8
        * X*Q = 48*48x8 > 8 
        * X*V = 48*48x8 > 8 
    * Create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
    * Z = V*AM = 8*8 > 8
    * ther FC layer called Out that:
    * Z*Out = 8*8x48 > 48
* Repeat this Ultimus block 4 times
* Then add final FC layer that converts 48 to 10 and sends it to the loss function.
* Model would look like this C>C>C>U>U>U>U>FFC>Loss
* Train the model for 24 epochs using the OCP that I wrote in class. Use ADAM as an optimizer. 
* Submit the link and answer the questions on the assignment page:
    * Share the link to the main repo (must have Assignment 7/8/9 model7/8/9.py files (or similarly named))
    * Share the code of model9.py
    * Copy and paste the Training Log
    * Copy and paste the training and validation loss chart

