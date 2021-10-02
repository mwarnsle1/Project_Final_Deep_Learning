# Final_Cap_Deep_Learning

# The Face Mask Identifier - Software for Infectious Disease Dept


# Overview

The science behind mask safety has certainly evolved since the onset of the pandemic. With the exception of the early months, masks have mostly been either recommended or required wear in public areas. As the mandates on mask-wearing continue to shift in the uncertain and fluctuating landscape of COVID-19, one aspect of the discussion is the proper and most effective way to wear a mask.

This topic is of increasing concern with the influx of more contagious variants, learned to be more airborn than previous versions of the virus. Data Scientists working in public health - particularly in environmental and/or epidemiological settings - often work collaboratively with professionals from other departments on projects to address issues like:

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

For my research, I obtained a public dataset from Kaggle, "Face Mask Detection", which consisted of three classes of image data inside a zip file; one for correct mask wearers, one for non-mask wearers, and the other for incorrect mask wearers.

They were then written to one large csv file. This was for both ease of accessibility and analysis of both email types. The combined dataset catalogued the categorized emails received by users. Before the model creation, I initially applied a number of exploratory analyses; such as first applying a label to the user column, which - was previously unlabeled - then, the code was further cleaned and the column values were confirmed. I then used one-hot encoding and added a key for the new target variable, so that the variable - which previously held strings in its column - could be fed properly into the model.

I applied some preliminary visualizations to check the initial distribution of the variables; by themselves and in relation to each other. I then proceeded to apply a progressive variation of models to observe their affect and accuracy. I applied a clustering method to the chosen features(DBSCAN), in order to display the categorization of the target variable. I also adjusted the parameters, applied hierarchical clustering, and a random forest technique to the dataset.

After observing the results, I applied another clustering method with TF-IDF and Kmeans to the data. I also applied some natural language processing techniques to the dataset. Specifically, word embedding and word vectorization were used to better predictive model and filter user emails. The results are discussed in the next section.


# Results

Due to the nature of the data(one-dimensional, few features etc), it was not initially conducive to clustering(DBSCAN) and dimensionality reduction and did not present any eventful results in the initial model, or random forest thereafter. However, after a number of additional features were created and another cluster model was attempted(TD-IDF + KMeans), better results were able to be obtained.

A predictive model was able to be created using natural language processing techniques. Using word embedding, a scatter plotting was used to visualize dots annotated with the words from the text. From the second NLP model, a predictive model was created using word vectorization; where each categorization was used to detect and filter new emails.


# Discussion & Recommendation

A closer look at the data indicates that Spam Detection/Spam Filters are best run on TD-IDF/KMeans clustering and NLP word vectorization models. While DBSCAN algorithm works well to find clusters of any shape and works better than hierarchical clustering, they perform better on more robust datasets.

Drawbacks of the dataset was having an uneven(higher) number of non-spam("Ham") emails; which may have negatively affected the predictive modeling in the random forest. This imbalance was overcome for the NLP model; however. Future iterations would still implement measures to balance out the dataset by under-sampling the "Ham" variable, and implement cross-validation with k-folds 5.
