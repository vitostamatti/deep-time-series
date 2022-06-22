
import tensorflow as tf
from tensorflow import keras


class Encoder(keras.layers.Layer):
    def __init__(self, lstm_units=[32,8], dropout=0.1):
        super(Encoder, self).__init__()

        model = keras.Sequential()
        for u in lstm_units[:-1]: 
            model.add(keras.layers.LSTM(units=u, activation='relu', return_sequences=True, dropout=dropout))

        model.add(keras.layers.LSTM(units=lstm_units[-1], activation='relu', return_sequences=False,dropout=dropout))
        
        self.model = model

    def call(self, x):
        out = self.model(x)
        return out


class Decoder(keras.layers.Layer):
    def __init__(self, features, timesteps, lstm_units=[8,32]):
        super(Decoder, self).__init__()

        model = keras.Sequential()
        model.add(keras.layers.RepeatVector(timesteps))
        for u in lstm_units:
            model.add(keras.layers.LSTM(units=u, return_sequences=True))

        model.add(keras.layers.TimeDistributed(keras.layers.Dense(features)))

        self.model = model

    def call(self, x):
        out = self.model(x)
        return out


class EncoderDecoder(keras.layers.Layer):
    def __init__(self, features, timesteps, lstm_units=[8,32], dropout=0.1, name=None):
        super(EncoderDecoder, self).__init__(name=name)
        self.encoder = Encoder(lstm_units=lstm_units,dropout=dropout)
        lstm_units.reverse()
        self.decoder = Decoder(features, timesteps, lstm_units=lstm_units)

    def call(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out