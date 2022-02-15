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
.
.
.
.
.
Number of correctly classified instances : 238 Total number of instances : 445
Accuracy : 0.534831460674157
