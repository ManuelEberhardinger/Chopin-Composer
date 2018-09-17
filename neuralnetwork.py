from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import pickle


class CustomModel:

    def __init__(self, network_input, network_output, notes, model_name, train):
        self.network_input = network_input
        self.network_output = network_output
        self.notes = notes
        self.n_vocab = len(set(notes))
        self.model_name = model_name
        self.train = train

        if self.train:
            self.model_path = "./weights/" + self.model_name + ".hdf5"
            self.model = getattr(self, self.model_name)()
        else:
            self.model_path = model_name
            self.model = load_model(self.model_path)

    def big_model(self):
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(2048))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def small_model2(self):
        model = Sequential()
        model.add(LSTM(
            128,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dense(2048))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def model_gru(self):
        model = Sequential()
        model.add(GRU(
            128,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(GRU(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(GRU(128))
        model.add(Dense(2048))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def small_model(self):
        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(256))
        model.add(Dense(2048))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def model(self):
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(GRU(512, activation='relu', dropout=0.3, recurrent_dropout=0.5, return_sequences=True))
        model.add(LSTM(512))
        model.add(Dense(2048))
        model.add(Dropout(0.3))
        model.add(Dense(2048))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def train_network(self, epochs, batch_size):
        if not self.train:
            return

        """ train the neural network """
        file_path = "./weights/" + self.model_name + "-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            file_path,
            monitor='loss',
            verbose=0,
            save_best_only=False,
            mode='min',
            period=50
        )
        callbacks_list = [checkpoint]

        history = self.model.fit(self.network_input, self.network_output, epochs=epochs,
                                 batch_size=batch_size, callbacks=callbacks_list)

        # save history of model
        with open("./" + self.model_name + "_history", 'wb') as f:
            pickle.dump(history.history, f)

        self.model.save(self.model_path)

        # model is trained so set flag to false
        self.train = False

    def get_int_to_note_mapping(self):
        # get all pitch names
        pitch_names = sorted(set(self.notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
        return int_to_note

    def generate_music(self, sequence_length):
        # in train mode so no music should be generated
        if self.train:
            return

        print("Start generating music..")
        start = np.random.randint(0, len(self.network_input) - 1)
        int_to_note = self.get_int_to_note_mapping()
        pattern = self.network_input[start]
        prediction_output = []

        # generate sequence_length notes
        for note_index in range(sequence_length):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)
            prediction = self.model.predict(prediction_input, verbose=1)
            index = int(np.argmax(prediction))
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

