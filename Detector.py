import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

#LOAD DATA
# The dataset contains 10015 images divided in two parts. 
# There 'HAM10000_metadata.csv' contains the ground truth
# of the dataset. Here we can use this file to retrieve 
# the images from the relative folders and also analyse what the images are.
path = '.'
file_name = os.path.join(path, 'HAM10000_metadata.csv')
df = pd.read_csv(file_name, na_values=['NA','?'])
disease_types = pd.unique(df['dx'])
print(disease_types)

# As you can see from the last print, there are 7 types of images
# classified in the dataset. Let's create a dictionary to better represent
# those diseases and then replace those with the full name of the patology.
images_type = {'bkl': 'Benign Keratosis',
                      'nv': 'Melanocytic Nevi',
                      'df': 'Dermatofibroma',
                      'mel': 'melanoma',
                      'vasc': 'Vascular Lesions',
                      'bcc': 'Basal Cell Carcinoma',
                      'akiec': "Bowen's disease"}
images_type
df.replace(images_type.keys(), images_type.values(), inplace=True)
df[:5]

# We can observe also that the ages range, varies between 0 and 80.
# This information will be useful in the final analysis 
# because we could tell wheter if the age is an increasing 
# factor of skin cancer or not.
age_range = pd.unique(df['age'])
age_range

# It is also possible to locate where different tumors where found in the 
# patients
cancers_location = pd.unique(df['localization'])
cancers_location

# The dataset has been developed during the last 20 years and 
# most of the images have been classified manually using a
# process that in medicine is called 'Histopathology' (histo) which entails
# to microscopically analyse a small portion of the skin tissue and then
# classifying it accordigly. There are also images which has been classified
# by the use of a tool called Cofocal Microscopy (cofocal) which
# allowed medicians to correctly identify where there's a disease
# and where it's not. Finally the data contains also a series of images 
# that have not been classified rigorously, denoted by 'follow_up'
# (data that needs follow-up examination) and 'consensus' data that has been
# classified by the consensus of a medician. Since these last three categories
# do not represent a rigorous result, we decided to exclude them 
# from the scope of the analysis.
classification_type = pd.unique(df['dx_type'])
print(classification_type)
# Drop the rows that have the 'consensus' or 'follow_up' value
# in the column 'dx_type'
indexes_consensus = df[df['dx_type'] == 'consensus'].index
indexes_follow_up = df[df['dx_type'] == 'follow_up'].index
df.drop(indexes_consensus, inplace=True)
df.drop(indexes_follow_up, inplace=True)
classification_type = pd.unique(df['dx_type'])
print(classification_type)

# Let's now check for NA values and columns that have NA values
df.isnull().any()
#Replace those NA values with the median of the values of that column
median = df['age'].median()
df['age'] = df['age'].fillna(median)
df.isnull().any()   

# In order to build a system capable to be trained and to give us results,
# it is crucial to encode the different types of skin cancer into 
# labels, to classify them. This is achieved using sklearn 
# by the following code.
encoder = preprocessing.LabelEncoder()
labels = encoder.fit_transform(df['dx'])
df['target'] = labels

# Now we would like to count the numbers of skin cancers according to each
# category to see if the dataset is well balanced or not.
path_part_1 = '/Users/tommasocapecchi/Datasets/HAM10000/Images'
dataset = []

# for image_name in df['image_id'][:10]:
#     image_path = os.path.join(path_part_1, image_name +'.jpg')
#     df['img_path'] = image_path
#     image = plt.imread(image_path, format='jpg')
#     dataset.append(image)
    