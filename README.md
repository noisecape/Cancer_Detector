# Cancer_Detector

Solution to the module INM702 of MSc Artificial Intelligence at City, University of London.

To run and execute the Notebook, the dataset can be downloaded at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T. There are three files that are **strictly required**:

1. HAM10000_images_part1.zip (5000 JPEG files)
2. HAM10000_images_part2.zip (5015 JPEG files)
3. HAM10000_metadata.csv

The first two files contain the actual images to be used to train and test the classifiers and can be downloaded by the above link. The third one is available in this repository and contains the ground truth values for each image as well as the features associated with them.

## Things to do before execute the code inside the Jupyter Notebook

1. Download the two folders' image from the link above.
2. Merge those images in one single folder called 'Images'
3. Ideally, you want the 'HAM10000_metadata.csv' file to be in the same directory as the 'Images' folder.
4. Open Jupyter Notebook and run the code. Dependancy python libraries you may want to install: Numpy, Keras, Scikit-learn, Pandas.

