from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score
from numpy import mean
import imblearn.metrics as im
import keras.wrappers.scikit_learn as wp
from sklearn.model_selection import GridSearchCV 
import numpy as np
import main_classification as mc
#def multi_layer_perceptron(samples_train, categorical_labels_train, samples_test, categorical_labels_test):
def multi_layer_perceptron( num_nodes=100 ,activation = 'relu' ,input_dim = 19):
    '''
    Implements a multi-layer perceptron used as a baseline for DL-networks evaluation
    '''
    
    print('Starting execution of single_layer_perceptron approach')
    # Total number of features


    input_dim = input_dim

    # Number of classes
    num_classes = 4

    #Number of neurons
    shallow_nodes_L1 = num_nodes
    shallow_nodes_L2 = num_nodes
    shallow_nodes_L3 = num_nodes

    # Define a Sequential model
    model = Sequential()

    # Build a multi-layer perceptron
    #Fully connected layers are defined using the Dense class.
    #Input layer
    model.add(Dense(shallow_nodes_L1, activation = activation, input_dim = input_dim))
    #Hidden layer
    model.add(Dense(shallow_nodes_L2, activation = activation))
    model.add(Dense(shallow_nodes_L3, activation = activation))
    #Output layer
    model.add(Dense(num_classes, activation = 'softmax'))
    # model.summary()

    print('single_layer_perceptron: Sequential model built')

    # Compile the model with categorical crossentropy loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Compile the model with categorical crossentropy loss function and Adam optimizer
    #model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print('single_layer_perceptron: Sequential model compiled with crossentropy loss function')
    # Plot DL architecture in PNG format
    #plot_model(model, to_file="single_layer_perceptron.png")
    return model
#def multi_layer_perceptron(samples_train, categorical_labels_train, samples_test, categorical_labels_test):

def train_test(samples_train, categorical_labels_train, samples_test, categorical_labels_test, num_nodes , activation): 
        # Define early stopping callback
    input_dim = samples_train.shape[1]

    # Number of classes
    num_classes = 4

    #Number of neurons
    shallow_nodes_L1 = num_nodes
    shallow_nodes_L2 = num_nodes
    shallow_nodes_L3 = num_nodes

    # Define a Sequential model
    model = Sequential()

    # Build a multi-layer perceptron
    #Fully connected layers are defined using the Dense class.
    #Input layer
    model.add(Dense(shallow_nodes_L1, activation = activation,  input_dim = input_dim))
    #Hidden layer
    model.add(Dense(shallow_nodes_L2, activation = activation))
    model.add(Dense(shallow_nodes_L3, activation = activation))
    #Output layer
    model.add(Dense(num_classes, activation = 'softmax'))
    # model.summary()

    print('single_layer_perceptron: Sequential model built')

    # Compile the model with categorical crossentropy loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Compile the model with categorical crossentropy loss function and Adam optimizer
    #model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print('single_layer_perceptron: Sequential model compiled with crossentropy loss function')
    earlystop = EarlyStopping(monitor='acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    # Training phase: train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes)
    train_model(model, 200, 50, callbacks_list, samples_train, categorical_labels_train, num_classes)
    print('single_layer_perceptron: Training phase completed')

    # Test with predict_classes: test_model(model, samples_test, categorical_labels_test)
    soft_values, predictions, training_soft_values, training_predictions, accuracy, fmeasure, macro_gmean, training_accuracy, training_fmeasure, training_macro_gmean = \
    test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test)
    print('single_layer_perceptron: Test phase completed')

    print('Ending execution of single_layer_perceptron approach')

    return soft_values, predictions, training_soft_values, training_predictions, accuracy, fmeasure, macro_gmean, training_accuracy, training_fmeasure, training_macro_gmean


def train_model(model, epochs, batch_size, callbacks, samples_train, categorical_labels_train, num_classes):
    '''
    Trains (fit) the keras model given as input
    '''

    one_hot_categorical_labels_train = to_categorical(categorical_labels_train, num_classes=num_classes)
    model.fit(samples_train, one_hot_categorical_labels_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)


def test_model(model, samples_train, categorical_labels_train, samples_test, categorical_labels_test):
    '''
    Test the keras model given as input with predict_classes()

    
    Epochs (nb_epoch) is the number of times that the model is exposed to the training dataset.
    Batch Size (batch_size) is the number of training instances shown to the model before a weight update is performed.
    '''

    predictions = model.predict_classes(samples_test, verbose=2)

    # Calculate soft predictionsmulti_layer_perceptron
    soft_values = model.predict(samples_test, verbose=2)
    mc.calculate_stats(predictions, categorical_labels_test , 'mlp_confusion_matrix', show_fig=False)

    training_predictions = model.predict_classes(samples_train, verbose=2)
    training_soft_values = model.predict(samples_train, verbose=2)

    # print(len(categorical_labels_test))
    # print(categorical_labels_test)

    # print(len(predictions))
    # print(predictions)

    # print(len(soft_values))
    # print(soft_values)

    # Accuracy, F-measure, and g-mean
    accuracy = accuracy_score(categorical_labels_test, predictions)
    fmeasure = f1_score(categorical_labels_test, predictions, average='macro')
    macro_gmean = mean(im.geometric_mean_score(categorical_labels_test, predictions, average=None))

    # Accuracy, F-measure, and g-mean on training set
    training_accuracy = accuracy_score(categorical_labels_train, training_predictions)
    training_fmeasure = f1_score(categorical_labels_train, training_predictions, average='macro')
    training_macro_gmean = mean(im.geometric_mean_score(categorical_labels_train, training_predictions, average=None))

    return soft_values, predictions, training_soft_values, training_predictions, accuracy, fmeasure, macro_gmean, training_accuracy, training_fmeasure, training_macro_gmean


def mlp_tuning(X_train , X_test , Y_train , Y_test):
    model = wp.KerasClassifier(build_fn = multi_layer_perceptron, input_dim = X_train.shape[1])

    grid_params = {'activation': ['relu' , 'sigmoid', 'tanh'] , 'num_nodes' : np.arange(10 , 120 , 10)}
    cv = GridSearchCV(estimator = model , param_grid = grid_params , n_jobs=-1 , verbose = 2 , cv = 5)
    grid_result = cv.fit(X_train , Y_train)
    print("Best: %f using %s" , (grid_result.best_params_))
    numnodes=grid_result.best_params_['num_nodes']
    activation=grid_result.best_params_['activation']
    soft_values, predictions, training_soft_values, training_predictions, accuracy, fmeasure, macro_gmean, training_accuracy, training_fmeasure, training_macro_gmean=train_test(X_train , Y_train , X_test , Y_test , num_nodes=numnodes, activation=activation)
    
    print(accuracy)
    return accuracy

    
#'hidden_layer_sizes':np.arange(50, 150)
