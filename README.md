# Project Final - Deep Learning

# The Face Mask Identifier - Machine-Learning for Public Health  


# Overview

The science behind mask safety has certainly evolved since the onset of the pandemic. With the exception of the early months, masks have mostly been either recommended or required wear in public areas. As the mandates on mask-wearing continue to shift in the uncertain and fluctuating landscape of COVID-19, one aspect of the discussion is the proper and most effective way to wear a mask.

This topic is of increasing concern with the influx of more contagious variants, learned to be more airborne than previous versions of the virus. Data Scientists working in public health - particularly in environmental and/or epidemiological settings - often work collaboratively with professionals from other departments on projects to address issues like:

- clean water and/or air quality testing
- mother-child/infant adverse risk prevention
- lead paint poisoning
- infectious disease spread


# PROPOSAL

A machine-learning algorithm is being created that can detect & identify incorrect mask-wearing in New York City for indoor establishments. This detection product will be marketed to establishments as part of the new "Keys to the City(all NYers having to be masked + show proof of vaccination)" program; especially in high-infection areas, based on COVID tracking (DOHMH) data.


# The Data

The data for this project consisted of one zipped folder; inside of which three folders of image data exist. Each image folder is labeled for each class it belongs to, and holds roughly 3000 images;

- 3000 for "with_mask"
- 2994 "without_mask"
- 2994 "mask_weared_incorrect"

Many datasets on mask detection come imbalanced and uncleaned; however, this dataset was adjusted (from an original dataset) so that each class had a similar distribution of images and noisy images that could be considered outliers were removed. Below, you can find the original and cleaned datasets:

Original datasets:
- https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
- https://www.kaggle.com/andrewmvd/face-mask-detection

Cleaned dataset:
- https://www.kaggle.com/vijaykumar1799/face-mask-detection


# Methods
# Initial Models

For my research, I obtained a public dataset from Kaggle, "Face Mask Detection", which consisted of three classes of image data inside a zip file; one for correct mask wearers, one for non-mask wearers, and the other for incorrect mask wearers.

They were then written to one large csv file and downloaded to my local drive. This was for both ease of accessibility and analysis of the image data. The large dataset catalogued the categorized images. Before the model creation, I preprocessed the data, mounted my Drive, unzipped the data, then applied a preliminary visual check, such as checking the labels/classes, and that the images could be read; individually and in a list. I then split and trained the data. The data was split - into ratios .09/.05/.05 for training, testing, and validation - using ImageDataGenerator, then stored in a numpy array. After reshaping the image data, I proceeded to build a convolutional neural network(CNN) model using the Sequential API.

I chose a CNN to build my model, as my specialization was in deep learning. Though the topic focused on the basics of artifical neural networks, further research explained that a simple CNN model could create better classification for image data. I created the current model with 16 layers using Conv2D, Batch normalization, Max Pooling, Dropout, Flatten and Dense layers.

- I began with a convolutional layer as the input layer, as this serves as the "feature map", whose information is fed to other layers to learn several other features of the input image
- A Batch normalization layer was used for the first hidden layer, to normalize the activations of the input volume before passing it into the next layer in the network...
   - followed by a pooling layer to also help reduce the spatial size of the input volume
- another round of layers were then added; with a convolutional layer, followed by a dropout layer to help prevent over-fitting, and a pooling layer
- this pattern is repeated three more times, twice increasing the neuron size
- a flattening layer was inserted to remove dimensionality and transform the input to 1D before transferring in to a Dense layer, where the neuron size is increased once more
- a final dropout layer is added before transferring it to the last layer, the output layer
For activation functions, I used ReLU, with the exception of the output layer, where softmax was used due to having its output being able to be read as probabilities in classification tasks.

The model was compiled using a fast optimizer, Adam, categorical crossentropy to minimize loss, and the accuracy metric to measure training performance, because it is a classification task and the dataset is balanced. The model was then trained using the `fit()` method.

The model was then evaluated using the `evaluate()` method of the model object to obtain the test set's accuracy using cross validation with KFolds, 5, to give the mean accuracy score of the value for the model. The recall, precision, and f1 scores were additionally checked to tell what portion of the values were correctly predicted, and which predictions were missed. I also used PyPlot line graphing and a box-and-whiskers plot to help visualize the results of the accuracy scores, and then finally, tested the model on a validation set before creating it for deployment

*Of note, I applied three versions of deep learning models with variations in layers and parameters, including this colab. The other two models' results are noted in the Results section.*

# Final Model

After the model's performance and results were thoroughly compared with Model A and C, Model B was chosen to undergo further analysis for mask predictions. To do that, a final model was built.

Another CNN deep learning model with parameters similar to Model B was created on unseen data - the test set - then saved to an H5 file, using the `save()` method after installing the h5py ibrary. Similar steps were taken - loading the dataset, scaling and reshaping the data, then defining, compiling, fitting, and evaluating the model.

After this, a new image was loaded and seen if the algorithm could detect its class.


# Results

As noted in the previous section, I preemptively created three sets of models, with adjusted parameters for comparison. However, due to the configuration of the third model, it was unable to run successfully without repeated crashing on my local device; thus, the scores for the first two models are compared. The links to the other two models are at the bottom of the section. The other models were run a total of seven times, with the values being reported being the average.

- Model Summaries:
   - Model A: 9-layer CNN model
      - 2 convolutional layers, 1 dropout, 2 pooling layers, 2 Dense layers, 1 Batch normalization and flattening layer
      - 0.9695 training data, with 0.7958 test accuracy.
   - Model B: 16 -layer CNN model
      - 4 convolutional, dropout, and pooling layers, 2 Dense, 1 Batch normalization and flattening layer
      - same activation functions, zero-padding added
      - 0.9068 training data, with 0.8648 test accuracy.
   - Model C: 7-layer model
      - 1 convolutional, 5 Dense layers, and 1 flattening layer
      - due to the model(i.e. its layers) being extremely imbalanced and not having the layers necessary to handle the data, it repeatedly crashed and could not be run. It would have needed much more alteration e.g. batch normalization, more flattening, and the adding in of other layers, e.g. MaxPool, Conv2D that would have made it virtually identical to the other versions
*Comparisons:*
   - Model A had a high training score, but a relatively lower test score, with a 0.174 difference between scores, which could indicate over-fitting
   - Model B had a lower training score, but a higher test score, with 0.0508 gap between sets
   - Model B's increased test accuracy indicates that this model was better at learning from the data
   - Model A's high training score combined with a larger gap between the sets indicates that its data was fit well, making it less likely to be generalized to the test set
   - the # of neurons were also adjusted(increased) in the second and third models; however, without the hidden layers to compensate for those adjustments and compute them appropriately in Model C, this quickly led to resource depletion.
   - In Model B, two additional layers - another convolutional layer with the same neurons as the last one(128), and a dense layer with 256 neurons - were added to the model. Here
   - Though further evaluation will be discussed, it seems apparent that for initial accuracy, the deeper, more complex model had higher test accuracy and as also less prone to over-fitting without causing errors.
- Model A https://colab.research.google.com/drive/1FdSYQvMVT1ob2zSygl0zw8gvdO300B7_?usp=sharing
- Model B https://colab.research.google.com/drive/100E-fDI_A4JKrtU8GbjLZQQ4iJ8dk4QD?usp=sharing
- Model C https://colab.research.google.com/drive/11ZbNiQeUZn7aCKMIFRiP6YWuXjKSanG8?usp=sharing

*Predictions:

- Further analyses conducted on the model's ability to predict the class of images resulted similar results for both Models A and B(e.g. 81% precision, 78% recall for Model A, 79% recall for Model B).

```
Precision(.81), Recall(.78), f1(.76)  - model A
Precision(.89), Recall(.88), f1(.88) - model B
```
While this isn't the highest accuracy we were striving for, it highlights the point that initial model accuracy isn't the pinnacle of truth, while definitely leaving room for improvement!

*Visualizations:*

- The results were visualized after being defined. A single figure with two subplots - one for loss and one for accuracy - were created showing model performance on the train and test set during each fold of the k-fold cross-validation. Blue lines indicated model performance on the training dataset and green lines indicated performance on the test dataset. The code below created the plot given the collected training histories for both models:

```
#Loss and Accuracy Learning Curves During k-Fold Cross-Validation:

#plot loss
plt.subplot(2, 1, 1)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='green', label='test')

#plot accuracy
plt.subplot(2, 1, 2)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='green', label='test')
plt.show()
```

Model B generally achieved a good fit, with train and test learning curves converging. A validation set was then used to test the model's ability to predict images.

- For image prediction, validation data was then loaded and split up, using a for loop and accessing the data directories. Using the predict() method on the model object, another for loop was created to find if the model could predict the labels on new data.

# Final Model

For the Model B, the same formula was used (e.g. layers, activation functions, batch size) to create the final model as had been used on the initial model. After running the test harness, the classification accuracy averaged to 74.459. This was noticably lower than the initial model on the validation set, which had a training and test accuracy of 90.68 and 86.48, respectively.

Finally, the saved model was then loaded into an image prediction function, using the `predict()` method for the model object, in order for it to be able to predict mask detection. Running the example prints the class for the instance(as well as the array the classes correspond to). As can be seen, the model's algorithm successfully loaded and prepared the image, loaded the model, and then correctly identified that the loaded image represented a person "without mask" - class 2.

# Discussion & Recommendation

A closer look at the data indicates that a deeper model(at least 16 layers) with the power needed for multi-class image processing contributes to better test accuracy and performance.

Though the dataset was large enough for variability, drawbacks of the dataset were being pre-aggregated and not having fresh generation. Additionally, accuracy dropping on the test set indicated that a possibly larger dataset to train on might have garnered a better score.

Though the models built were mildly successful at building a predictive deep learning model, improvements could be made in the model configuration that could result in the overall upgrade. One would be improving the learning algorithm, and the other would be increasing the depth of the model.

Regarding the learning algorithm, there are a number of aspects that could be improved upon, namely aspects that could affect the rate of learning. Already mentioned previously was batch normalization. By standardizing the outputs, it both has the effects of changing the distribution of the output and stabilizing and accelerating the learning process. Model B, which has the most layers, initially did not have this layer/process, but added it after further research and consultation on its effects(also after the model had similar destabilizing returns as Model C. In addition to this, the images could have been converted to one-channel/grayscale images during the preprocessing phase, in order to reduce the weights of the model and make the process faster. This was not initially thought of due to the nature of the project, but was learned during the research and can be implemented in future iterations(before launch).

Increasing, or even exploring, the model depths and/or capacity is another way to conduct model improvement. This could be done in a similar pattern as Model B by initially adding on convolutional and pooling layers with the same number of neurons and adjusting the parameters(e.g. batch size, activation functions, padding), then gradually increasing the parameters, as well as the neurons. Layers can continue to be added on this way while observing the change(or no change) in scores. As the comparisons between Models B and A showed, adding complexity(in layers) to the model can add to the accuracy, as well.
