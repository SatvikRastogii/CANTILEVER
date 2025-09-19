import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import sys
from preprocessing import load_and_preprocess_data

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Parameters ---
BATCH_SIZE = 64
EPOCHS = 100  # Increased for better results
LATENT_DIM = 512  # Increased for better model capacity
VOCAB_SIZE = 30000  # Increased vocabulary size
MAX_SEQUENCE_LENGTH = 150  # Increased sequence length
EMBEDDING_DIM = 256  # Embedding dimension
DROPOUT_RATE = 0.3  # Dropout for regularization

def create_tokenizer(input_texts, target_texts, vocab_size=VOCAB_SIZE):
    """Create and fit tokenizer on the combined texts."""
    print("Creating tokenizer...")
    tokenizer = Tokenizer(
        num_words=vocab_size, 
        oov_token='<unk>',
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    )
    
    # Fit on both input and target texts
    all_texts = input_texts + target_texts
    tokenizer.fit_on_texts(all_texts)
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Number of words: {tokenizer.num_words}")
    
    return tokenizer

def prepare_data(input_texts, target_texts, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """Prepare data for training."""
    print("Preparing training data...")
    
    # Convert texts to sequences
    encoder_input_seq = tokenizer.texts_to_sequences(input_texts)
    decoder_input_seq = tokenizer.texts_to_sequences(target_texts)
    
    # Pad sequences
    encoder_input_data = pad_sequences(
        encoder_input_seq, 
        maxlen=max_length, 
        padding='post',
        truncating='post'
    )
    decoder_input_data = pad_sequences(
        decoder_input_seq, 
        maxlen=max_length, 
        padding='post',
        truncating='post'
    )
    
    # Create decoder target data (one-hot encoded)
    vocab_size = len(tokenizer.word_index) + 1
    decoder_target_data = np.zeros(
        (len(target_texts), max_length, vocab_size), 
        dtype='float32'
    )
    
    for i, seq in enumerate(decoder_input_seq):
        for t, word_index in enumerate(seq):
            if t > 0 and word_index < vocab_size:  # Skip the <start> token
                decoder_target_data[i, t - 1, word_index] = 1.0
    
    print(f"Encoder input shape: {encoder_input_data.shape}")
    print(f"Decoder input shape: {decoder_input_data.shape}")
    print(f"Decoder target shape: {decoder_target_data.shape}")
    
    return encoder_input_data, decoder_input_data, decoder_target_data

def build_model(vocab_size, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM, 
                max_length=MAX_SEQUENCE_LENGTH, dropout_rate=DROPOUT_RATE):
    """Build the sequence-to-sequence model with attention."""
    print("Building the model...")
    
    # --- Encoder ---
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(
        vocab_size, 
        embedding_dim, 
        mask_zero=True,
        name='encoder_embedding'
    )(encoder_inputs)
    
    # Encoder LSTM with dropout
    encoder_lstm = LSTM(
        latent_dim, 
        return_state=True, 
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name='encoder_lstm'
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # --- Decoder ---
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(
        vocab_size, 
        embedding_dim, 
        mask_zero=True,
        name='decoder_embedding'
    )(decoder_inputs)
    
    # Decoder LSTM with dropout
    decoder_lstm = LSTM(
        latent_dim, 
        return_sequences=True, 
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Attention mechanism
    attention = Attention(name='attention')
    context_vector = attention([decoder_outputs, encoder_outputs])
    
    # Concatenate context vector with decoder output
    decoder_concat = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
    
    # Dense layer with dropout
    decoder_dense = Dense(
        latent_dim, 
        activation='relu',
        name='decoder_dense'
    )(decoder_concat)
    decoder_dropout = Dropout(dropout_rate)(decoder_dense)
    
    # Output layer
    decoder_outputs = Dense(
        vocab_size, 
        activation='softmax',
        name='decoder_outputs'
    )(decoder_dropout)
    
    # --- Training Model ---
    model = Model(
        [encoder_inputs, decoder_inputs], 
        decoder_outputs,
        name='seq2seq_model'
    )
    
    # Compile model
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense

def build_inference_models(encoder_inputs, encoder_states, decoder_inputs, 
                          decoder_embedding, decoder_lstm, decoder_dense, 
                          vocab_size, embedding_dim=EMBEDDING_DIM, latent_dim=LATENT_DIM):
    """Build inference models for encoder and decoder."""
    print("Building inference models...")
    
    # --- Encoder Inference Model ---
    encoder_model = Model(encoder_inputs, encoder_states, name='encoder_inference')
    
    # --- Decoder Inference Model ---
    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Decoder embedding
    decoder_emb2 = decoder_embedding(decoder_inputs)
    
    # Decoder LSTM
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        decoder_emb2, 
        initial_state=decoder_states_inputs
    )
    decoder_states2 = [state_h2, state_c2]
    
    # Dense layer
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    
    # Output layer
    decoder_outputs2 = Dense(vocab_size, activation='softmax')(decoder_outputs2)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2,
        name='decoder_inference'
    )
    
    return encoder_model, decoder_model

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, 
                epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2):
    """Train the model with callbacks."""
    print("Training the model...")
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'saved_model/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        [encoder_input_data, decoder_input_data[:, :-1]], 
        decoder_target_data[:, 1:, :],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_models_and_artifacts(encoder_model, decoder_model, tokenizer, 
                             vocab_size, max_length, latent_dim, embedding_dim):
    """Save all model artifacts."""
    print("Saving model artifacts...")
    
    # Create saved_model directory if it doesn't exist
    os.makedirs('saved_model', exist_ok=True)
    
    # Save models
    encoder_model.save('saved_model/encoder_model.h5')
    decoder_model.save('saved_model/decoder_model.h5')
    
    # Save tokenizer
    tokenizer_json = tokenizer.to_json()
    with open('saved_model/tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    
    # Save parameters
    params = {
        'vocab_size': vocab_size,
        'max_sequence_length': max_length,
        'latent_dim': latent_dim,
        'embedding_dim': embedding_dim,
        'word_index': tokenizer.word_index
    }
    
    with open('saved_model/params.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    
    print("Model artifacts saved successfully!")

def main():
    """Main training function."""
    print("Starting Aura chatbot training...")
    
    # Check if dataset exists
    dataset_path = 'data/aura_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run create_dataset.py first to generate the dataset.")
        return
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    input_texts, target_texts = load_and_preprocess_data(dataset_path)
    
    if not input_texts or not target_texts:
        print("No data loaded. Please check your dataset.")
        return
    
    print(f"Loaded {len(input_texts)} conversation pairs")
    
    # Create tokenizer
    tokenizer = create_tokenizer(input_texts, target_texts)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Prepare data
    encoder_input_data, decoder_input_data, decoder_target_data = prepare_data(
        input_texts, target_texts, tokenizer
    )
    
    # Build model
    model, encoder_inputs, encoder_states, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense = build_model(
        vocab_size
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    history = train_model(
        model, 
        encoder_input_data, 
        decoder_input_data, 
        decoder_target_data
    )
    
    # Build inference models
    encoder_model, decoder_model = build_inference_models(
        encoder_inputs, encoder_states, decoder_inputs, 
        decoder_embedding, decoder_lstm, decoder_dense, vocab_size
    )
    
    # Save models and artifacts
    save_models_and_artifacts(
        encoder_model, decoder_model, tokenizer, 
        vocab_size, MAX_SEQUENCE_LENGTH, LATENT_DIM, EMBEDDING_DIM
    )
    
    print("Training completed successfully!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == "__main__":
    main()
