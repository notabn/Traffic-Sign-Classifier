#**Traffic Sign Recognition** 

---

### README

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./hist_class.jpg "Class distribution"
[image1]: ./hist_insight.jpg "Visualization"
[image2]: ./grayscale.png "Grayscaling"
[image3]: ./class_aug.jpg "Augmentation Noise"
[image4]: traffic_internet/3.jpg "Traffic Sign 1"
[image5]: ./traffic_internet/5.png "Traffic Sign 2"
[image6]: traffic_internet/17.png "Traffic Sign 3"
[image7]: ./traffic_internet/12.png "Traffic Sign 4"
[image8]: ./traffic_internet/22.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---



###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over different 43 classes. The class distribution is not uniform and some classes
are better represented than others. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Preprocessing 

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because shape and color intensity are good features to classify signs.

Less color channels means less dimensions and for a small network as LeNet it can be an advantage.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I then normalized the image data regarding that neural networks work much better with smaller values.
As a last step, to improve contrast images I performed a histogram equalisation.


####2. Data


My final training set had 51999 number of images. My validation set and test set had 4410 and 12630 number of images.

The forth code cell of the IPython notebook contains the code for augmenting the data set taken from . I decided to generate additional data to improve the network capabilities to generalize. 
To add more data to the the data set, I rotated, translated and sheared the image.  

The difference between the original data set and the augmented data set is the following :

![alt text][image3]




####3. Network Architecture

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU                  |
| Max Pooling           | 2x2 stride, outputs 16x16x32
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64				    | 
| Convolution 5x5	    | 1x1 stride, same padding, outputs 8x8x128
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128		     									|
| Fully connected		| outputs 2048
| RELU					|	 
| DROPOUT
| Fully connected		| outputs 1024
| RELU	
| DROPOUT	      
| Fully connected		| outputs 512 
| RELU
| DROPOUT	  			|
| Softmax				| outputs 43      									|



####4. Network Training

The code for training the model is located in the tenth cell of the ipython notebook. 

To train the model, I used the LeNet architecture and tuned the parameters as follows:

*  varied the learning rate with 0.008 and 0.005, the validation accuracy was no higher that 93%
*  decreased the learning rate to 0.001 and simultaneously increase the number of epoch to 40, the validation accuracy increased to 96% as below

    * Validation Loss = 0.201
    * Validation Accuracy = 0.965    
    * Test accuracy = 0.936
    
* keep learning rate at 0.001  and increase number of epoch to 50 ; after epoch 45 the loss was oscillating around 96% 

    * Validation Loss = 0.296
    * Validation Accuracy = 0.961
    * Test accuracy = 0.94

* with learning rate at 0.001  and 70 epoch the model is performs as follows

    * Validation Loss = 0.501
    * Validation Accuracy = 0.955
    * Test accuracy = 0.934

* decreasing the batch size to 50  improved the accuracy with 1% in the test 
    * Validation Loss = 0.568
    * Validation Accuracy = 0.964
    * Test accuracy = 0.943
    but on the new images I had an accuracy only of 20%

* add another convolution layer:

    * Validation Loss = 0.259
    * Validation Accuracy = 0.968
    * Test accuracy = 0.944

* with data augmentation with data size 51999 the test accuracy 







####5. Solution

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

   * Validation Loss = 0.259
   * Validation Accuracy = 0.968
   * Test accuracy = 0.94
    

I tried the LeNet Architecture with two convolution layers and two fully connected layer. This type of architecture performs well on the MNIST dataset.
A test accuracy of 94% was achieved with LeNet after image preprocessing the images tunning the learning rate and number of epochs.


To improve the test accuracy I started training with augmented data. I further added dropouts with a probability of 0.9
in the last convolutional layer and to the fully connected layers when training to further generalize. 


###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

                  Image       Prediction
         Priority road       Ahead only
              No entry         No entry
            Bumpy road       Bumpy road
    Speed limit (60km/h)      No vehicles
    Speed limit (80km/h)  Traffic signals


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This doesn't compare to the accuracy on the test set of 93.4%

####3. Softmax probabilities

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the no entry and bumpy road signs the model is sure of the prediction  with aprobability of 1%. The the top five probabilities

* No Entry

              Prediction   Probability
                No entry  1.000000e+00
    Roundabout mandatory  4.669926e-08
        Turn right ahead  2.168699e-09
           Priority road  7.213173e-10
                    Stop  6.321932e-11
                    

* Bumpy road

                  Prediction   Probability
                 Bumpy road  1.000000e+00
          Bicycles crossing  3.876817e-08
                  Road work  1.776608e-10
            Traffic signals  1.099153e-10
        Road narrows on the right  3.592186e-11
  
  
  The model is not able to recognize the following sign for which it calculated the following top five soft max probabilities 

  Priority sign
  
              Prediction  Probability
              Ahead only     0.490671
    Speed limit (60km/h)     0.462833
         General caution     0.022604
             No vehicles     0.006507
                   Yield     0.005151
                 
                 
  Speed limit (60km/h) 3
  
        Prediction   Probability
        No vehicles  9.999992e-01
               Stop  8.475606e-07
          Keep left  1.574779e-10
    General caution  8.545054e-11
         Keep right  2.466549e-11
       
  Speed limit (80km/h) 5
  
                              Prediction      Probability
                          Traffic signals     0.977379
                     Speed limit (60km/h)     0.012178
                     Roundabout mandatory     0.010305
    Right-of-way at the next intersection     0.000129
                      Speed limit (80km/h)    0.000005