#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[sample_images]: ./images_for_submission/sample_images.png "Randomly chosen sample images from the training set."
[visualization]: ./images_for_submission/visualization.png "Visualization of the training set."
[after_grayscale]: ./images_for_submission/after_grayscale.png "Images after grayscaling."
[after_normalization]: ./images_for_submission/after_normalization.png "Images after normalization."
[augmented]: ./images_for_submission/augmented.png "Sample image after augmented."
[web_images]: ./images_for_submission/web_images.png "Web Images that I found on web."

[LeNet_Sermanet_Lecun]: ./images_for_submission/LeNet_Sermanet_Lecun.jpeg "Sermanet Lecun Article Architecture"

[without_augmented_data]: ./images_for_submission/without_augmented_data.tiff "Sermanet Lecun Article Architecture without augmented images"

[just_training_augmented]: ./images_for_submission/just_training_augmented.tiff "Sermanet Lecun Article Architecture just training data augmented"

[training+validation_augmented]: ./images_for_submission/training+validation_augmented.tiff "Sermanet Lecun Article Architecture training and validation data augmented"

[validation_splitted_from_training]: ./images_for_submission/validation_splitted_from_training.tiff "Sermanet Lecun Article Architecture training and validation data augmented"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

First I selected 10 random images to see some of the training images: 
![sample images][sample_images]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data uniformly distributed in the training set:

![visualization of the training data][visualization]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the article of Sermanet and Lecun specified that grayscaling images prfromed better results. 

Here is the example of the same random images above after grayscaling.

![Images after grayscale][after_grayscale]

As a last step, I normalized the image data because it was suggested in the lessons and it was very easy to apply to the set. I also plotted the images after normalization to check how they look like and surprisingly saw that they are looking the same as previously grayscaled images. Here the same image set after normalization:

![Images after normalization][after_normalization]

I decided to generate additional data because Sermanet and Lecun 's article says that creating the augmented data also increase the models performance. 

To add more data to the the data set, I used the same techniques applied in the article:

1. First I applied "Random Translation by 2 pixels height and width
1. Second I applied "Random Scaling between 0.9 and 1.1
1. And finallay I applied "Random rotation between -15 and + 15 degrees.    

I augmented whole trainig set before appyling grayscaling and normalization and I splitted the augmented set into training_augmented set and validation_augmented set aand after that merged the sets. I have chosen 4500 of the augmented set after shuffled them in order to create some randomness on the set for the validation and kept the rest for adding to training data. 

Here the example of an original image and an augmented image:

![Image after augmented][augmented]

The difference between the original data set and the augmented data set is the following:
Before Augment:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410


After Augment:

* Total Training size: 65098  labels size: 65098
* Total Test size: 12630  labels size: 12630
* Total Validation size: 8910  labels size: 8910


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         			| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 							|
| RELU						|												|
| Max pooling	      		| 2x2 stride, 28x28x6 in, outputs 14x14x6 							|
| Convolution 5x5	   	| 1x1 stride, valid padding, 14x14x6 in, outputs 10x10x16   		   		|
| RELU						|       										|
| Max pooling				| 2x2 stride, 10x10x16 in, outputs 5x5x16      							|
| Convolution 5x5		| 1x1 stride, valid padding, 5x5x6 in, outputs 1x1x400					|
|	RELU 					| No Max Pooling After this layer we flatten this layer and combine it with flattened Layer2							|
| Flatten 1x1x400 		| Input 1x1x400, outputs 400
| Flatten 5x5x16	     	| Input 5x5x16, outputs 400
| Concatenate				| 400 + 400, outputs 800
| Dropout	layer			|
| Fully Connected		| Input 800, outputs 43
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer as it is used in the LeNet lab, I decided to keep (as far as I remembered from lessons or some other sources it performs much better on traffic signs, Lenet models etc.) it as my optimizer. My final settings were as the following:

* batch size: 128
* epochs: 60
* learning rate: 0.0005
* mu: 0
* sigma: 0.1
* dropout keep probability: 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 1.000
* validation set accuracy of 0.963 
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The first architecture I tried is standart Lenet architecture. The accuracy was around 0.89 as said in the project notes. So, I read the Sermanet and Lecun article and decided to try to apply it for the solution.

![LeNet_Sermanet_Lecun Architecture][LeNet_Sermanet_Lecun]

First I tried the model with only grayscaled and normalized images and the result was below 0.93 but it was increased compared to standart Lenet model. The plotted training graph was like below:

 ![LeNet_Sermanet_Lecun Architecture without augmented data][without_augmented_data]

I played several parameters, like learning rate, dropout probability, number of epocs etc. And also tried augmenting the data. Here below the result is after augmenting just training data:

 ![LeNet_Sermanet_Lecun Architecture Just Training Augmented][just_training_augmented]

After augmenting the training set the validation accuracy increased 2% and I thought that if I apply the augmentation the validation set the acuuracy would increase but it did not happened as I expected. The accuracy was dropped. Here's the results:

 ![LeNet_Sermanet_Lecun Architecture Training and Validation Augmented][training+validation_augmented]

Finally I tried to choose validation data from augmented training set. And the best results achieved. Here's the final results:

 ![LeNet_Sermanet_Lecun Architecture Validation Spliited Augmented][validation_splitted_from_training]


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Images downloaded from web][web_images]

The first image might be difficuFlt to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)		| Speed limit (60km/h)   								| 
| Right-of-way at the next intersection     								| Right-of-way at the next intersection 								|
| Speed limit (30km/h)		| Speed limit (30km/h)								|
| Priority road	      		| Priority road					 			|
| General caution			| General caution     								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.4%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability => Label    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		=> 3			| Speed limit (60km/h)   										| 
| 4.14263276e-11    => 2   		| Speed limit (50km/h)										|
| 2.93211427e-11		=>	5			| Speed limit (80km/h)										|
| 2.55662106e-11	   => 25      	| Road work				 						|
| 3.30055753e-12		=> 11	    	| Right-of-way at the next intersection      										|


For the second image : 

| Probability => Label    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		=> 11			| Right-of-way at the next intersection    											| 
| 1.18109168e-11    => 30   		| Beware of ice/snow										|
| 7.71171955e-20		=>	26			| Traffic signals 										|
| 1.23086277e-20	   =>  18      	| General caution					 					|
| 2.62493883e-22		=> 28	    	| Children crossing       										|


For the third image : 

| Probability => Label    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		=> 1			| Speed limit (30km/h)   										| 
| 3.24956611e-20    => 0   		| Speed limit (20km/h)										|
| 2.47219599e-20		=>	2			| Speed limit (50km/h)										|
| 1.10454158e-22	   =>  5      	| Speed limit (80km/h)					 					|
| 4.73766317e-26		=> 40	    	| Roundabout mandatory      										|
    
For the fourth image : 

| Probability => Label    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		=> 12			| Priority road  										| 
| 2.11854681e-19    => 35   		| Turn right ahead										|
| 3.33225380e-20		=>	40			| Roundabout mandatory										|
| 1.20704528e-22	   =>  3      	| Speed limit (60km/h)					 					|
| 3.18050289e-23		=> 15	    	| No vehicles      										|
             
For the fifth image : 

| Probability => Label    	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         		=> 18			| General caution   										| 
| 1.66616574e-22    => 26   		| Traffic signals 										|
| 1.27612719e-26		=>	27			| Pedestrians 										|
| 3.98527568e-30	   => 11     		| Right-of-way at the next intersection				 			|
| 5.47299523e-36		=> 28	    	| Children crossing      										|
 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


