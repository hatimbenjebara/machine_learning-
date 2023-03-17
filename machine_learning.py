import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
cols =["flength","fwidth","fsize","fconc","fconc1","fasym","fm3long","fm3trans","falpha","fdist","class"]
data= pd.read_csv("magic04.data", names=cols)
print(data.head())
#let's understand data. each row is one sample item in our data. in each of row or sample the one quality 
# for each or one value for each of column and one of them is class. 
#now we are going to predict for future whether the class is g for gamma or H for Hadron this classification
#other variables or columns are knows as features . we are oing to pass into our model in order to help us 
#predict the lable, which in this case is the "class" column.
print(data["class"].unique()) # we notice that we have string in cols class 
data["class"] = (data["class"]=="g").astype(int) # convert string to int 
print(data.head()) # now is good 
for label in cols[:-1]: # check if the columns a part column =class have in relation with class 
    # inside the data get everything where class=1 the label column
    plt.hist(data[data["class"]==1][label], color="red", label= 'gamma', alpha=0.7, density=True) 
    plt.hist(data[data["class"]==0][label], color="blue", label= 'hardon', alpha=0.7, density=True) 
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
# we notice at fig(proba,fasym) there's symetrie 
#create our train, validation, test data
train, valid, test= np.split(data.sample(frac=1),[int(0.6*len(data)), int(0.8*len(data))])
#where to split in 0.6 time the length of data. so and then cast that 10 integer thats going to be first
#the first place where you know, i cut it off and that'll be my training data
# it's means that our training data between 60% and 80%
def scale_data(data):
    x = data[ data.columns[:-1]].values
    y = data[ data.columns[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)# take x and fit he standard scalar to x and then transform all those values
    #create all data 
    data_1=np.hstack((x,np.reshape(y,(-1,1))))# take an array and horizontally stack them together mean put them side by side
    #numpy is sentive about dimension here we have x 2d and y 1 d.for that we will reshape y to 2 d 
    #where the negative one just means infer what this dimension value would be which ends up being the length of y
    return data_1, x, y
print(train)
print(len(train))
print(len(train[train["class"]==1]))
print(len(train[train["class"]==0]))
train, x_train, y_train = scale_data(train)
valid, x_valid, y_valid = scale_data(valid)
test, x_test, y_test = scale_data(test)
print(len(x_train))
print(sum(y_train==1))
#first model : K-nearest neighbors 
#first thing define a euclidien distances 
knn_model = KNeighborsClassifier()
print(knn_model.fit(x_train,y_train))
y_pred = knn_model.predict(x_test)
print(y_pred)
print(classification_report(y_test,y_pred))
#accuracy is 0.83 which is actually pretty good, mean, what;s closest to we get actually 83% accuracy which mean 
#how many do we get right. per centage of precision
#model 2 : naive bayes 
nb_model = GaussianNB()
nb_model = nb_model.fit(x_train,y_train)
y_pred = nb_model.predict(x_test)
print(classification_report(y_test,y_pred))
#model 3 : logistic regression 
lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train,y_train)
y_pred = lg_model.predict(x_test)
print(classification_report(y_test,y_pred))
#model 4 : support vector machines (svm)
svm_model = SVC()
svm_model = nb_model.fit(x_train,y_train)
y_pred = svm_model.predict(x_test)
print(classification_report(y_test,y_pred))
#model 5: tensorflow
def plot_history(history):
    fig,(ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'],label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('binary crossentropy')
    ax1.grid(True)
    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    plt.show()
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
def train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1)])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss= 'binary_crossentropy', metrics= ['accuracy'])
    history = nn_model.fit( x_train , y_train , epochs=epochs,batch_size=batch_size, validation_split=0.2)
    return nn_model, history
least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [16,32,64]:
    for dropout_prob in [0,0.2]:
        for lr in [0.1,0.005, 0.001]:
            for batch_size in [32,64,128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, lr{lr}, batch_size{batch_size}")
                model, history = train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_loss(history)
                val_loss= model.evaluate(x_valid,y_valid)[0]
                plot_accuracy(history)
                plot_history(history)
                if val_loss < least_val_loss:
                    least_val_loss=val_loss
                    least_loss_model = model
y_pred = least_loss_model.predict(x_test)
print(y_pred)
y_pred= (y_pred>0.5).astype(int).reshape(-1,)
print(y_pred)
print(classification_report(y_test,y_pred))
