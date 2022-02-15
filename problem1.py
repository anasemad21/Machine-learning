
import pandas as pd
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

def read_file(path):
    file = pd.read_csv(path,sep=',', header=None)
    return file
data=read_file('house-votes-84.data')
def hanle_missing_values(data):
    #first loop on columns
    #seconde loop on rows
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[i][j] == "?":
                #we replaced missing value with most frequent value
                x = data[i].value_counts().idxmax() # to get most frequent value
                data[i][j] = x
    return data
data=hanle_missing_values(data)
def naming_attributes(data):
    data.columns = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing',
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                    'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                    'mx-missile', 'mx-missile', 'synfuels-corporation-cutback', 'education-spending',
                    'education-spending', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'
                    ]
    return data
data=naming_attributes(data)
def compute_accuracy(replaced_ytest,predicted):
    correct_predictions = 0
    # iterate over each label and check
    for true, predicted in zip(replaced_ytest, predicted):
        if true == predicted:
            correct_predictions += 1
    # compute the accuracy
    accuracy = correct_predictions/len(replaced_ytest)*100
    return accuracy
#function take the data and build decisionTree model based on training size parameter
def decisionTree(data,train_size):
    mean=0
    list_accuracy=[]
    #open file to write accuracy report for each run with diffrent train size
    f = open('results.txt', 'a')
    #to write train size in file
    f.write(f'{train_size}\n')
    max_accuracy=0 # to store max accuracy to plot its model
    best_decisionTree=None  #initialize dummy model
    #we iterate three times with random splites
    for i in range(3):
        #split data into train and test based on train size parameter
        train, test = train_test_split(data, train_size=train_size)
        #training set
        y_train=train['Class Name']
        x_train = train.iloc[:, 1:]
        #testing set
        y_test=test['Class Name']
        x_test=test.iloc[:,1:]
        #converting data into integers
        replaced_xTrain=x_train.replace(('n','y'), (-1,1))
        replaced_ytrain=y_train.replace(('democrat', 'republican'), (1, 0))
        replaced_xTest=x_test.replace(('n','y'), (-1,1))
        replaced_ytest=y_test.replace(('democrat', 'republican'), (1, 0))
        #create object of classifier
        decisionTree=tree.DecisionTreeClassifier()
        #build and train the decisionTree model
        decisionTree=decisionTree.fit(replaced_xTrain,replaced_ytrain)
        #testing the model with test set
        predicted=decisionTree.predict(replaced_xTest)
        accuracy=compute_accuracy(replaced_ytest,predicted)
        list_accuracy.append(accuracy)
        #get the max accuracy of three times of  run
        if accuracy > max_accuracy:
            max_accuracy=accuracy
            best_decisionTree = decisionTree
        #write accuracy of each run in the file
        f.write(f'{accuracy}\n')
    num_nodes=decisionTree.tree_.node_count
    mean=sum(list_accuracy)/3
    _min=min(list_accuracy)
    _max=max(list_accuracy)
    f.write('Max: ')
    f.write(f'{_max}\n')
    f.write('Min: ')
    f.write(f'{_min}\n')
    f.write('Mean: ')
    f.write(f'{mean}\n')
    f.close()
    return max_accuracy,best_decisionTree,num_nodes

#collect accuracies to plot on xAxis
list_accuracy=[]

#collect number of nodes to plot on xAxis
list_num_nodes=[]
list_training_size=[0.4,0.5,0.6,0.7,0.8]
#collect max accuracies to retrieve the best model and visualize it
list_max_accuracies=[]
#list of decisionTree models to extract the best one and visualize
list_best_tree=[]

# to clear the file because of appending mode
f = open('results.txt', 'w')
f.write('')
f.close()


# here we call decisionTree function with different training size
for i in list_training_size:
    max_accuracy,best_decisionTree,num_nodes=decisionTree(data,train_size=i)
    list_accuracy.append(max_accuracy)
    list_best_tree.append(best_decisionTree)
    list_num_nodes.append(num_nodes)

max_value = max(list_accuracy)
max_index = list_accuracy.index(max_value)
best_tree=list_best_tree[max_index]


plt.plot(list_training_size,list_accuracy)
plt.xlabel('Training')
plt.ylabel('Accuracy')
plt.title('Training Size VS Accuracy')
plt.show()

plt.plot(list_training_size,list_num_nodes)
plt.xlabel('Training')
plt.ylabel('Number Of Nodes')
plt.title('Training Size VS Number Of Nodes')
plt.show()

tree_graph=export_graphviz(best_tree,out_file=None,filled=True,rounded=True,
                           special_characters=True,feature_names=['handicapped-infants', 'water-project-cost-sharing',
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                    'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                    'mx-missile', 'mx-missile', 'synfuels-corporation-cutback', 'education-spending',
                    'education-spending', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
                           ,class_names=['democrat ','republican'])
graph=graphviz.Source(tree_graph)
graph.render('best_Tree_Visualisation')
