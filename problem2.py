
def read_file(path):
    #file = pd.read_excel(path, header=None)
    file = pd.read_csv(path, header=None)
    return file

def getkMinIndex(distance, k):
    min_indexes = []
    for i in range(k):
        min_value = min(distance)
        min_index = distance.index(min_value)
        min_indexes.append(min_index)
        del (distance[min_index])
    return min_indexes

def getClasses(min_indexes, data_train):
    classes = []
    for i in range(len(min_indexes)):
        classes.append(data_train[min_indexes[i]][data_train.shape[1] - 1])
    return classes


def getmostfrequency(classes, data_train):
    predicted_value = 0
    frequency = []
    l = []
    unique_class = np.unique(classes)
    for i in unique_class:
        frequency.append(classes.count(i))
    maxfreq = max(frequency)
    train_classess_file = list(data_train[:,data_train.shape[1]-1])
    if frequency.count(maxfreq) > 1:
        # there is tie
        list_max_freq_indexes = []
        max_classes_freq = []
        file_indexies = []
        for k in range(len(frequency)):
            if frequency[k] == maxfreq:
                list_max_freq_indexes.append(k)  # indexes of the maxiumim classes
        for l in list_max_freq_indexes:
            max_classes_freq.append(unique_class[l])
        for i in max_classes_freq:
            file_indexies.append(train_classess_file.index(i))
        first_occur = min(file_indexies)
        predicted_value=train_classess_file[first_occur]
        #index_predicted = file_indexies.index(first_occur)
        #predicted_value = max_classes_freq[index_predicted]
    # no tie
    else:
        index_max_freq = frequency.index(maxfreq)
        predicted_value = unique_class[index_max_freq]
    return predicted_value


def compute_accuracy(replaced_ytest, predicted):
    correct_predictions = 0
    # iterate over each label and check
    for true, predicted in zip(replaced_ytest, predicted):
        print("predicted", predicted, " ", "actual", true)
        if true == predicted:
            correct_predictions += 1


    # compute the accuracy
    accuracy = correct_predictions / len(replaced_ytest) * 100
    print("Number of correctly classified instances:",correct_predictions,"Total number of instances : ",len(replaced_ytest))
    return accuracy



def knn (k,data_train,data_test):
        inference_classes = list(data_test[:,data_test.shape[1]-1])
        predicted = []
        for j in range(len(data_test)):
            distance = []
            min_indexes = []
            classes=[]
            list_values=data_test[j]
            for l in range(len(data_train)):
                    temp=0
                    for m in range((data_train.shape[1]-1)):
                        temp+=math.pow(list_values[m]-data_train[l][m],2)
                    distance.append(math.sqrt(temp))
            min_indexes=getkMinIndex(distance,k)
            classes=getClasses(min_indexes,data_train)
            predicted.append(getmostfrequency(classes,data_train))
        accuracy=compute_accuracy(inference_classes,predicted)
        return accuracy

data_train = read_file("trainingSet.csv")
train = np.array(data_train.values)
data_test = read_file("TestSet.csv")
test = np.array(data_test.values)


for i in range(1,10):
    result = knn(i, train, test)
    print(result)
    print("k",i)
