# -*- coding: utf-8 -*-

from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class MLP():
    def __init__(self,
                 l1_out=512, l2_out=512, l3_out=512,
                 l1_drop=0.2, l2_drop=0.2, l3_drop=0.2,
                 bn1=0, bn2=0, bn3=0,
                 batch_size=100,
                 epochs=10,
                 validation_split=0.1):
        self.l1_out, self.l1_drop, self.bn1 = l1_out, l1_drop, bn1
        self.l2_out, self.l2_drop, self.bn2 = l2_out, l2_drop, bn2
        self.l3_out, self.l3_drop, self.bn3 = l3_out, l3_drop, bn3
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.load()
        self.__model = self.model()

    def load(self):


        return 1, 2, 3, 4

    def model(self):
        model = Sequential()

        model.add(Dense(self.l1_out, input_shape=(1, 100,)))
        if self.bn1 == 0:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.l1_drop))

        model.add(Dense(self.l2_out))
        if self.bn2 == 0:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.l2_drop))

        model.add(Dense(self.l3_out))
        if self.bn3 == 0:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.l3_drop))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model

    def fit(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)

        self.__model.fit(self.__x_train, self.__y_train,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         verbose=0,
                         validation_split=self.validation_split,
                         callbacks=[early_stopping])

    def evaluate(self):
        self.fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        return evaluation
