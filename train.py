import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, AveragePooling1D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

from dataset_preparation import Datatset
from setup_dataframe import Setup


def build_model(in_shape):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=6, strides=1, padding='same', activation='relu', input_shape=(in_shape, 1)))
    model.add(AveragePooling1D(pool_size=4, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=4, strides=2, padding='same'))

    model.add(Conv1D(128, kernel_size=6, strides=1, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=4, strides=2, padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=6, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=4, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_build_summary(mod_dim, tr_features, val_features, val_labels):
    model = build_model(mod_dim)
    model.summary()

    score = model.evaluate(val_features, val_labels, verbose=1)
    accuracy = 100 * score[1]

    return model


if __name__ == '__main__':
    print("Get dataframes")
    dp = Datatset()
    females, males = dp.create_dfs()
    print(females.head())
    print(males.head())

    setup = Setup()
    print("Extracting features")
    female_X, female_y = setup.Xysplit(females)
    male_X, male_y = setup.Xysplit(males)

    Females_Features = setup.setup_dataframe('Female', female_X, female_y)
    Males_Features = setup.setup_dataframe('Male', male_X, male_y)

    encoder = OneHotEncoder()

    female_Y = encoder.fit_transform(np.array(female_y).reshape(-1, 1)).toarray()
    male_Y = encoder.fit_transform(np.array(male_y).reshape(-1, 1)).toarray()

    nogender_X = np.concatenate((female_X, male_X))
    nogender_Y = np.concatenate((female_Y, male_Y))

    # nogender

    x_train, x_test, y_train, y_test = train_test_split(nogender_X,
                                                        nogender_Y,
                                                        random_state=42,
                                                        test_size=0.20,
                                                        shuffle=True)

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    # females
    x_trainF, x_testF, y_trainF, y_testF = train_test_split(female_X, female_Y, random_state=0, test_size=0.20,
                                                            shuffle=True)
    x_trainF = np.expand_dims(x_trainF, axis=2)
    x_testF = np.expand_dims(x_testF, axis=2)

    # males
    x_trainM, x_testM, y_trainM, y_testM = train_test_split(male_X, male_Y, random_state=0, test_size=0.20,
                                                            shuffle=True)
    x_trainM = np.expand_dims(x_trainM, axis=2)
    x_testM = np.expand_dims(x_testM, axis=2)

    # model initialization
    total_model = model_build_summary(x_train.shape[1], x_train, x_test, y_test)
    female_model = model_build_summary(x_trainF.shape[1], x_trainF, x_testF, y_testF)
    male_model = model_build_summary(x_trainM.shape[1], x_trainM, x_testM, y_testM)

    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=4, min_lr=0.000001)
    batch_size = 32
    n_epochs = 75

    # model training
    history = total_model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs,
                              validation_data=(x_test, y_test), callbacks=[rlrp])

    female_history = female_model.fit(x_trainF, y_trainF, batch_size=batch_size, epochs=n_epochs,
                                      validation_data=(x_testF, y_testF), callbacks=[rlrp])

    male_history = male_model.fit(x_trainM, y_trainM, batch_size=batch_size, epochs=n_epochs,
                                  validation_data=(x_testM, y_testM), callbacks=[rlrp])

    # genderless
    score = total_model.evaluate(x_train, y_train, verbose=0)
    print("Mixed-gender emotions training Accuracy: {0:.2%}".format(score[1]))

    score = total_model.evaluate(x_test, y_test, verbose=0)
    print("Mixed-gender emotions testing Accuracy: {0:.2%}".format(score[1]))

    total_model.save("saved_models/total_model.h5")

    # female
    score = female_model.evaluate(x_trainF, y_trainF, verbose=0)
    print("Female emotions training Accuracy: {0:.2%}".format(score[1]))

    score = female_model.evaluate(x_testF, y_testF, verbose=0)
    print("Female emotions testing Accuracy: {0:.2%}".format(score[1]))

    female_model.save("saved_models/female_model.h5")

    # male
    score = male_model.evaluate(x_trainM, y_trainM, verbose=0)
    print("Male emotions training Accuracy: {0:.2%}".format(score[1]))

    score = male_model.evaluate(x_testM, y_testM, verbose=0)
    print("Male emotions testing Accuracy: {0:.2%}".format(score[1]))

    male_model.save("saved_models/male_model.h5")

    # predicting on test data.
    pred_test = female_model.predict(x_testF)
    y_pred = encoder.inverse_transform(pred_test)
    y_test_ = encoder.inverse_transform(y_testF)

    # print confusion matrix for FEM
    cm = confusion_matrix(y_test_, y_pred)
    cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
    print(cm)

    # predicting on test data.
    pred_test = male_model.predict(x_testM)
    y_pred = encoder.inverse_transform(pred_test)
    y_test_ = encoder.inverse_transform(y_testM)

    # print confusion matrix for MAL
    cm = confusion_matrix(y_test_, y_pred)
    cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
    print(cm)
