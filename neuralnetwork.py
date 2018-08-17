from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import numpy
import os


class CustomModel:

    def __init__(self, network_input, network_output, notes):
        self.network_input = network_input
        self.network_output = network_output
        self.notes = notes
        self.n_vocab = len(set(notes))

        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        self.weightsPath = 'weights.hdf5'
        self.model = model
        self.model_trained = False

    def train(self, epochs, batch_size):
        """ train the neural network """
        file_path = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            file_path,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        self.model.fit(self.network_input, self.network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
        self.model.save_weights(self.weightsPath)
        self.model_trained = True

    def get_int_to_note_mapping(self):
        # get all pitch names
        pitch_names = sorted(set(self.notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
        return int_to_note

    def generate_music(self, sequence_length):
        if not os.path.isfile(self.weightsPath) or not self.model_trained:
            print("Model not trainend and no weigths file exists.. return..")
            return

        self.model.load_weights(self.weightsPath)

        start = numpy.random.randint(0, len(self.network_input) - 1)
        int_to_note = self.get_int_to_note_mapping()
        pattern = self.network_input[start]
        prediction_output = []

        # generate #sequence_length notes
        for note_index in range(sequence_length):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)
            prediction = self.model.predict(prediction_input, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        return prediction_output
