from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score
from numpy import mean
import imblearn.metrics as im
 
def multi_layer_perceptron(samples_train, categorical_labels_train, samples_test, categorical_labels_test):
    '''
    Implements a multi-layer perceptron used as a baseline for DL-networks evaluation
    '''

    print('Starting execution of single_layer_perceptron approach')
    # Total number of features
    input_dim = samples_train.shape[1]

    # Number of classes
    num_classes = 4

    #Number of neurons
    shallow_nodes_L1 = 100
    shallow_nodes_L2 = 100
    shallow_nodes_L3 = 100

    # Define a Sequential model
    model = Sequential()

    # Build a multi-layer perceptron
    #Fully connected layers are defined using the Dense class.
    #Input layer
    model.add(Dense(shallow_nodes_L1, activation = 'relu', input_dim = input_dim))
    #Hidden layer
    model.add(Dense(shallow_nodes_L2, activation = 'softmax'))
    model.add(Dense(shallow_nodes_L3, activation = 'softmax'))
    #Output layer
    model.add(Dense(num_classes, activation = 'softmax'))
    # model.summary()

    print('single_layer_perceptron: Sequential model built')

    # Compile the model with categorical crossentropy loss function
    # model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])

    # Compile the model with categorical crossentropy loss function and Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print('single_layer_perceptron: Sequential model compiled with crossentropy loss function')

    # Plot DL architecture in PNG format
    #plot_model(model, to_file="single_layer_perceptron.png")

    # Define early stopping callback
    earlystop = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
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

    # Calculate soft predictions
    soft_values = model.predict(samples_test, verbose=2)

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
