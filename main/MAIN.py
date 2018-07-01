import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential, model_from_yaml
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from masterthesis.nowe.AccuracyHistory import AccuracyHistory
from masterthesis.nowe.image import load_original_images, load_test_data

BATCH_SIZE = 10
EPOCHS = 1000


def learn():
    [x_train, y_train, num_classes] = load_original_images(True)
    [x_test, y_test, n] = load_original_images(False)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    history = AccuracyHistory()

    model = Sequential()
    # model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
    # model.add(Dropout(0.2))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    # model.add(Dense(5000, activation='relu'))
    model.add(Conv1D(32, kernel_size=5, strides=2,
                     activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 10, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1500, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.squared_hinge,
                  optimizer=keras.optimizers.SGD(lr=0.002),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[history])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("hello")

    plt.plot(range(1, EPOCHS + 1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    save(model)


def predict():
    model = load_model()

    [x_train, y_train, num_classes] = load_test_data()
    c = model.predict_classes(x_train)
    c = c.astype(np.int8)
    acc = accuracy(c, y_train)
    print(acc)

    print('class 0')
    exp0 = np.copy(y_train)
    pred0 = np.copy(c)

    exp0[exp0 != 0] = -1
    exp0[exp0 == 0] = 1
    pred0[pred0 != 0] = -1
    pred0[pred0 == 0] = 1
    print("PRECISION " + str(precision_score(exp0, pred0, average='binary')))
    print("RECALL " + str(recall_score(exp0, pred0, average='binary')))
    print("ACCURACY " + str(accuracy_score(exp0, pred0)))
    print("F1SCORE " + str(f1_score(exp0, pred0, average='binary')))
    print('')

    print('class 1')
    exp1 = np.copy(y_train)
    pred1 = np.copy(c)

    exp1[exp1 != 1] = -1
    exp1[exp1 == 1] = 1
    pred1[pred1 != 1] = -1
    pred1[pred1 == 1] = 1
    print("PRECISION " + str(precision_score(exp1, pred1, average='binary')))
    print("RECALL " + str(recall_score(exp1, pred1, average='binary')))
    print("ACCURACY " + str(accuracy_score(exp1, pred1)))
    print("F1SCORE " + str(f1_score(exp1, pred1, average='binary')))
    print('')

    print('class 2')
    exp2 = np.copy(y_train)
    pred2 = np.copy(c)

    exp2[exp2 != 2] = -1
    exp2[exp2 == 2] = 1
    pred2[pred2 != 2] = -1
    pred2[pred2 == 2] = 1
    print("PRECISION " + str(precision_score(exp2, pred2, average='binary')))
    print("RECALL " + str(recall_score(exp2, pred2, average='binary')))
    print("ACCURACY " + str(accuracy_score(exp2, pred2)))
    print("F1SCORE " + str(f1_score(exp2, pred2, average='binary')))
    print('')

    print('class 3')
    exp3 = np.copy(y_train)
    pred3 = np.copy(c)

    exp3[exp3 != 3] = -1
    exp3[exp3 == 3] = 1
    pred3[pred3 != 3] = -1
    pred3[pred3 == 3] = 1
    print("PRECISION " + str(precision_score(exp3, pred3, average='binary')))
    print("RECALL " + str(recall_score(exp3, pred3, average='binary')))
    print("ACCURACY " + str(accuracy_score(exp3, pred3)))
    print("F1SCORE " + str(f1_score(exp3, pred3, average='binary')))
    print('')

    print('class 4')
    exp4 = np.copy(y_train)
    pred4 = np.copy(c)

    exp4[exp4 != 4] = -1
    exp4[exp4 == 4] = 1
    pred4[pred4 != 4] = -1
    pred4[pred4 == 4] = 1
    print("PRECISION " + str(precision_score(exp4, pred4, average='binary')))
    print("RECALL " + str(recall_score(exp4, pred4, average='binary')))
    print("ACCURACY " + str(accuracy_score(exp4, pred4)))
    print("F1SCORE " + str(f1_score(exp4, pred4, 0.73, average='binary')))

    print("PRECISION " + str(precision_score(y_train, c, average='micro')))
    print("RECALL " + str(recall_score(y_train, c, average='micro')))
    print("ACCURACY " + str(accuracy_score(y_train, c)))
    print("F1SCORE " + str(f1_score(y_train, c, average='micro')))


def save(model):
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    yaml_file = open('../model/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("../model/model.h5")
    print("Loaded model from disk")
    return loaded_model


def accuracy(predictions, labels):
    right_predictions = 0

    for i in range(0, len(predictions)):
        if predictions[i] == labels[i]:
            right_predictions = right_predictions + 1

    return right_predictions / len(predictions)


predict()
