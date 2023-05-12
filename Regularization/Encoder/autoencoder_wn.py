from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import numpy as np

class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Definir a entrada
        self.input_layer = Input(shape=(self.input_dim,))
        
        # Criar a camada oculta
        self.hidden_layer = Dense(self.encoding_dim, activation='relu')(self.input_layer)
        
        # Criar a camada de saída
        self.output_layer = Dense(self.input_dim, activation='sigmoid')(self.hidden_layer)
        
        # Criar o modelo
        self.model = Model(self.input_layer, self.output_layer)
        
    def train(self, x_train, epochs, batch_size):
        # Compilar o modelo
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Adicionar ruído gaussiano às entradas
        x_train_noisy = x_train + np.random.normal(loc=0.0, scale=0.1, size=x_train.shape)
        
        # Treinar o modelo
        self.model.fit(x_train_noisy, x_train, epochs=epochs, batch_size=batch_size)
        
    def reconstruct(self, x):
        # Reconstruir as entradas
        x_reconstructed = self.model.predict(x)
        return x_reconstructed