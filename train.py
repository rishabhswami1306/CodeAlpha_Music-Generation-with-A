import os
import pickle
import argparse
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from model import create_lstm_model

class StatusCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        
    def on_train_begin(self, logs=None):
        self._write_status({
            "status": "training",
            "epoch": 0,
            "total_epochs": self.total_epochs,
            "loss": 0.0,
            "accuracy": 0.0
        })
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._write_status({
            "status": "training",
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "loss": float(logs.get('loss', 0.0)),
            "accuracy": float(logs.get('accuracy', 0.0)),
            "val_loss": float(logs.get('val_loss', 0.0)) if 'val_loss' in logs else None,
            "val_accuracy": float(logs.get('val_accuracy', 0.0)) if 'val_accuracy' in logs else None
        })
        
    def _write_status(self, data):
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/training_status.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error writing status callback: {e}")

def train_pipeline(epochs=20, batch_size=64, use_embedding=True):
    """
    Loads preprocessed MIDI data, instantiates the LSTM model,
    compiles it, and trains it with checkpoint and early stopping callbacks.
    """
    print("\n--- STEP 4: Training ---")
    data_path = "data/preprocessed_data.pkl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}. Please run preprocess.py first.")
        
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    vocab_size = data['vocab_size']
    sequence_length = data['sequence_length']
    y = data['y']
    
    # Load input format depending on embedding configuration
    if use_embedding:
        X = data['X']
        print(f"Training WITH token embedding. Input shape: {X.shape}")
    else:
        X = data['X_normalized']
        print(f"Training WITHOUT token embedding (normalized input). Input shape: {X.shape}")
        
    print(f"Output shape (one-hot classes): {y.shape}")
    
    # Create the model
    model = create_lstm_model(vocab_size, sequence_length, use_embedding=use_embedding)
    model.summary()
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Setup directories
    os.makedirs("models", exist_ok=True)
    
    # Define callbacks
    checkpoint_path = "models/best_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    status_callback = StatusCallback(epochs)
    callbacks_list = [checkpoint, early_stop, status_callback]
    
    # Train the model
    print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        validation_split=0.1,  # use 10% of sequences for validation
        verbose=1
    )
    
    # Save the final model weights and complete model file
    final_path = "models/final_music_model.h5"
    model.save(final_path)
    print(f"Training complete. Best model checkpoint saved to '{checkpoint_path}'. Final model saved to '{final_path}'.")
    
    # Save completion status
    try:
        with open("data/training_status.json", "w") as f:
            json.dump({
                "status": "completed",
                "epoch": epochs,
                "total_epochs": epochs,
                "model_path": final_path
            }, f)
    except Exception as e:
        print(f"Error writing completion status: {e}")
        
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Music Generator")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--no_embedding', action='store_true', help="Disable embedding layer (use direct normalized float inputs)")
    
    args = parser.parse_args()
    
    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_embedding=not args.no_embedding
    )
