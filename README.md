# Machine-learning

#### ##### promlem 1
You will use this data to learn a decision tree that predicts the political party of the representative 
based on his /her vote .
Dataset link : https://archive.ics.uci.edu/ml/datasets/congressional+voting+records
Use the voting data to train a decision tree to predict political party (Democrat or Republican) 
based on the voting record. Use 25% of the members of congress for training and the rest for 
testing. Rerun this experiment three times and notice the impact of different random splits of the 
data into training and test sets. 
Report the sizes and accuracies of these trees in each experiment
• Measure the impact of training set size on the accuracy and the size of the learned
tree. Consider training set sizes in the range (30-70%). 
Because of the high variance due to random splits repeat the experiment with five different 
random seeds for each training set size then report the mean, maximum and minimum accuracies 
at each training set size. 
Also measure the mean, max and min tree size.
● Start with training data size 40% , 50% .... Until you reach 80%.
● Turn in two plots showing how accuracy varies with training set size and how the number of 
nodes in the final tree varies with training set size.
● The data set contained many missing values , i.e., votes in which a member of congress failed 
to participate. To solve those issue insert—for each absent vote—the voting decision of the 
majority.

### ##### promlem 2
Implement your own simple KNN classifier using python, (Don’t use any build in
functions)
Orange Restricted
● Use provided train and test file yeast_train.txt,yeast_test.txt
(http://vlm1.uta.edu/~athitsos/courses/cse4309_fall2020/assignments/uci_datasets/)
● Each record in dataset contain feature values are separated by commas, and the last value on 
each line is the class label
● If there is a tie in the class predicted by the k -nearest neighbors, then
among the classes that have the same number of votes, the tie should be broken in favor of the 
class comes first in the Train file.
● Use Euclidean distance to compute distances between instances.
● Report accuracy on testing data when k=1,2,3....9.
● As output, your programs should print the value of k used for the test
set on the first line, each output line should list the predicted class
label, and actual class label.
● Also output the number of correctly classified test instances, and the
total number of instances in the test set &Accuracy.
Example :
k value : 3
Predicted class : POX Actual class : CYT.

Number of correctly classified instances : 238 Total number of instances : 445
Accuracy : 0.5348314606741573



## problem 3

Sign Language to Speech:
About
The data set is a collection of images of alphabets from the American Sign Language, separated 
in 29 folders which represent the various classes.
Content
The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of 
which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
These 3 classes are very helpful in real-time applications, and classification.
The test data set contains a mere 29 images, to encourage the use of real-world test images.
Dataset link : https://archive.ics.uci.edu/ml/datasets/congressional+voting+records
• Train 3 different classifiers to classify the 29 classes [You can use sklearn].
• Use different input for training (RGB , GREY Binary)
• Report Precision & recall for each experiment.
## continue problem 3
• Train 2 different CNN architectures to classify the 29 classes [You can use Keras].
• Use RGB input for training 
• Report Precision & recall for each experiment

### problem 4
Anomaly detection – Feature Selection
• Dataset:https://datasetsearch.research.google.com/search?query=Breast%20Cancer%20D
ataset&docid=L2cvMTFqOWM3ejY5Yw%3D%3D
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast 
mass. They describe characteristics of the cell nuclei present in the image. n the 3-
dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear 
Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and 
Software 1, 1992, 23-34].
Given the attached dataset of Breast cancer prediction, apply the following:
Orange Restricted
o Detailed-illustration in a report for the applied techniques during the feature 
selection,
pre-processind.
o Evaluate on 3 different model and report precision - recall
