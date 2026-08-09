from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dropout, Dense, Embedding, Input, 
    RepeatVector, TimeDistributed, Bidirectional
)

def create_lstm_model(vocab_size, sequence_length, use_embedding=True):
    """
    Builds and compiles an LSTM-based sequence model in TensorFlow/Keras.
    Can be configured to use either a token embedding layer or direct raw sequence input.
    """
    model = Sequential()
    
    if use_embedding:
        # Embedding layer: maps integer index sequence to dense vectors of shape (batch, seq_len, embed_dim)
        model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=sequence_length))
        # Stacked LSTM layers with dropout to prevent overfitting
        model.add(LSTM(512, return_sequences=True))
    else:
        # Direct input layer: expects normalized numerical inputs of shape (batch, seq_len, features)
        model.add(LSTM(512, input_shape=(sequence_length, 1), return_sequences=True))
        
    model.add(Dropout(0.3))
    
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    
    # Dense projection layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Softmax output over vocabulary representing note/chord probabilities
    model.add(Dense(vocab_size, activation='softmax'))
    
    return model

# --- Extended Section: LSTM-Based GAN (Generative Adversarial Network) ---

def build_gan_generator(latent_dim, sequence_length, vocab_size):
    """
    LSTM-based Generator for the GAN structure.
    Takes a 1D random noise vector (latent space) and projects it into a sequence 
    of note/chord probability distributions.
    """
    noise = Input(shape=(latent_dim,))
    
    # Project latent noise and repeat across sequence timesteps
    x = Dense(256, activation='relu')(noise)
    x = RepeatVector(sequence_length)(x)
    
    # LSTM Layers process the temporal sequence
    x = LSTM(512, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(512, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    
    # TimeDistributed layer applies Dense projection to each timestep independently
    generated_sequence = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)
    
    model = Model(noise, generated_sequence, name="Generator")
    return model

def build_gan_discriminator(sequence_length, vocab_size):
    """
    LSTM-based Discriminator for the GAN structure.
    Takes a sequence of note/chord distributions (real or generated) and classifies 
    them as real (1.0) or fake/generated (0.0).
    """
    sequence_input = Input(shape=(sequence_length, vocab_size))
    
    x = LSTM(512, return_sequences=True)(sequence_input)
    x = Dropout(0.3)(x)
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    validity = Dense(1, activation='sigmoid')(x)
    
    model = Model(sequence_input, validity, name="Discriminator")
    return model

def build_complete_gan(generator, discriminator, latent_dim):
    """
    Combines generator and discriminator into a single GAN network.
    Freezes discriminator weights during generator updates.
    """
    # Freeze discriminator
    discriminator.trainable = False
    
    # Complete adversarial path
    noise_input = Input(shape=(latent_dim,))
    generated_seq = generator(noise_input)
    validity = discriminator(generated_seq)
    
    gan_model = Model(noise_input, validity, name="LSTM-GAN")
    return gan_model
