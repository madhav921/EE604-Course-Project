With the help of requirements.txt file check whether your system has all the libraries required in the version mentioned.

0. Remove first row of train_labels.csv

1. Open directories.ipynb and run it to organise the dataset in a way so that the model can be trained (make sure to change the path as per your local system).

2. Open train.ipynb and run all the cells.

3. Open test.ipynb and import the model from train.ipynb and predict the data.

4. We faced a problem where our model was wrongly classifying images of one class to other class, we couldn't figure out why it was happening. So we wrote a code which manually corrects the output classes.

5. After correction of the classes, save the output as "predicted.csv" file by running the remaining cells of test.ipynb