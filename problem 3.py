import cv2
from cv2 import *
import keras
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
ef knn(x_train_kn,y_train_kn,x_test_kn,y_test_kn):
    n = KNeighborsClassifier(n_neighbors=9)
    n.fit(x_train_kn, y_train_kn)
    y_predict_kn = n.predict(x_test_kn)
    accuracy = accuracy_score(y_test_kn, y_predict_kn)
    print("Accuracy: ",accuracy*100)
    pre=precision_score(y_test_kn,y_predict_kn,average="micro")
    rec = recall_score(y_test_kn, y_predict_kn, average='micro')
    measure = 2 * (pre * rec) / (pre + rec)
    print("recall: ", rec*100)
    print("precision: ", pre*100)
    print("Measure: ", measure)

def Tree(x_train_ds,y_train_ds,x_test_ds, y_test_ds):
    decisionTree = tree.DecisionTreeClassifier()
    # build and train the decisionTree model
    decisionTree = decisionTree.fit(x_train_ds, y_train_ds)
    # testing the model with test set
    y_predict_ds = decisionTree.predict(x_test_ds)
    #print(y_predict_ds)
    print("Accuracy: ",accuracy_score(y_test_ds, y_predict_ds) * 100)
    pre = precision_score(y_test_ds, y_predict_ds, average="micro")
    rec = recall_score(y_test_ds, y_predict_ds, average='micro')
    measure = 2 * (pre * rec) / (pre + rec)
    print("recall: ", rec * 100)
    print("precision: ", pre * 100)
    print("Measure: ", measure)
def logistic (X_train,y_train, X_test, y_test ):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    pre = precision_score(y_test,y_predict,average="micro")
    rec = recall_score(y_test, y_predict,average="micro")
    print("Accuracy: ",accuracy_score(y_test, y_predict)*100)
    measure = 2 * (pre * rec) / (pre + rec)
    print("recall: ", rec * 100)
    print("precision: ", pre * 100)
    print("Measure: ", measure)
    #print("result",y_predict)

path="D:\\Bioinformatics\\Fourth-Year\\Machine_Learning_And_Bioinformatics\\assignmt_2\\ASL_Alphabet_Dataset\\asl_alphabet_train"
files_train=os.listdir(path)
size=65
path2="D:\\Bioinformatics\\Fourth-Year\\Machine_Learning_And_Bioinformatics\\assignmt_2\ASL_Alphabet_Dataset\\asl_alphabet_test"
file_test=os.listdir(path2)
def read_image_test(file,path):
    x_test_list=[]
    y_test_list=[]
    for j in file:
        image = (cv2.imread(path + '\\' + j))
        image = cv2.resize(image, (size, size))
        y_test_list.append(j)
        x_test_list.append(image)
    for i in range(len(y_test_list)):
        y_test_list[i] = y_test_list[i].split("_")[0]
    print("reading test train finished ")
    return x_test_list,y_test_list
def read_image_train(file,path):
    x_train_list=[]
    y_train_list=[]
    for i in file:
        counter=0
        file_img = os.listdir(path+'\\'+i)
        temp_path=path+'\\'+i
        for j in range(0,len(file_img),3):
              img=(cv2.imread(temp_path+'\\'+file_img[j]))
              img = cv2.resize(img, (size, size))
              #img=img.flatten()
              x_train_list.append(img)
              y_train_list.append(i)
             # counter+=1
              #if(counter==10):
               #  break
    print("reading train finished ")
    return x_train_list , y_train_list
def image_proccesing(images):
    canny_list = []
    for i in range(len(images)):
        img=images[i]
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(image=img_blur, threshold1=4, threshold2=115)
        canny_list.append(edges.flatten())
    canny= np.array(canny_list)
    return canny


def main():

    flatten_train=[]# to store RGB train_images after flatten
    gray_images = [] #to store copy of images as gray
    binary_images = [] #to store copy of images as binary
    flatten_test=[] # to store the test images after flatten it and convert it to gray to be one channel
    x_train, y_train = read_image_train(files_train, path)
    x_test, y_test = read_image_test(file_test, path2)


    #converting copy of images into gray
    # for i in range(len(x_train)):
    #     img=(cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY))
    #     img=img.flatten()
    #     gray_images.append(img)

    #converting copy of images into binary based on the gray copy
    # for i in range(len(gray_images)):
    #     r, threshold = cv2.threshold(gray_images[i], 100, 255, cv2.THRESH_BINARY)
    #     binary_images.append(threshold)
    # cv2.imshow("kjfkjad",binary_images[1])
    # cv2.imshow("kj",gray_images[1])

    # flatten images
    for i in range(len(x_train)):
        flatten_train.append(x_train[i].flatten())

    #call image proccesing on the train images
    #cv2.imshow("addjvkj", binary_images[1])
    #canny_flatten_list=image_proccesing(binary_images)

    #converting the test images into gray and flatten it
    # for i in x_test:
    #     img=(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
    #     flatten_test.append(img.flatten())


    # flatten the test images to run with RGB
    for i in x_test:
        flatten_test.append(i.flatten())


    #calling different classifier

    #knn(flatten_train,y_train,flatten_test,y_test)
    #Tree(flatten_train,y_train,flatten_test,y_test)
    logistic(flatten_train,y_train,flatten_test,y_test)


main()


cv2.waitKey(0)
cv2.destroyAllWindows()
