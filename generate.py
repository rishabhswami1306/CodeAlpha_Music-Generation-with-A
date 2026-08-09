import os
import pickle
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument

def sample_with_temperature(prediction_probabilities, temperature=0.7):
    """
    Applies temperature scaling to prediction probabilities and samples an index.
    A lower temperature makes the model more conservative/predictable.
    A higher temperature makes it more creative/random.
    """
    if temperature <= 0.0:
        return np.argmax(prediction_probabilities)
        
    predictions = np.log(prediction_probabilities + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    prediction_probabilities = exp_preds / np.sum(exp_preds)
    
    # Sample from the adjusted distribution
    return np.random.choice(len(prediction_probabilities), p=prediction_probabilities)

def generate_music(model_path="models/best_model.h5", output_name="generated_music", 
                   num_notes=100, temperature=0.7, use_embedding=True):
    """
    Generates a MIDI file of classical music using the trained LSTM model.
    """
    print("\n--- STEP 5: Music Generation ---")
    data_path = "data/preprocessed_data.pkl"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Preprocessed data file not found. Run preprocess.py first.")
        
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    int_to_note = data['int_to_note']
    note_to_int = data['note_to_int']
    vocab_size = data['vocab_size']
    sequence_length = data['sequence_length']
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Specified model path '{model_path}' not found, attempting to load 'models/final_music_model.h5'...")
        model_path = "models/final_music_model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found. Run train.py first.")
            
    print(f"Loading trained model from '{model_path}'...")
    model = load_model(model_path)
    
    # Choose a random seed sequence from the training data
    if use_embedding:
        X = data['X']
    else:
        X = data['X_normalized']
        
    random_index = np.random.randint(0, len(X) - 1)
    seed_sequence = X[random_index]
    
    print(f"Seeding generation with a random sequence of length {sequence_length}...")
    
    # Keep track of prediction output
    prediction_output = []
    
    # Initial sequence state
    current_sequence = seed_sequence.copy()
    
    # Generate note sequences iteratively
    for note_index in range(num_notes):
        # Format input for prediction
        if use_embedding:
            # Shape: (1, sequence_length)
            prediction_input = np.reshape(current_sequence, (1, sequence_length))
        else:
            # Shape: (1, sequence_length, 1)
            prediction_input = np.reshape(current_sequence, (1, sequence_length, 1))
            # If not using embedding, data was normalized during preprocessing
            # Let's check: if current_sequence is indices, we normalize it
            # The seed sequence is already normalized if use_embedding is False
            # During loop updates, we insert the normalized value
            pass
            
        # Predict next note
        prediction = model.predict(prediction_input, verbose=0)[0]
        
        # Sample with temperature
        idx = sample_with_temperature(prediction, temperature)
        
        # Map back to note/chord string
        predicted_note = int_to_note[idx]
        prediction_output.append(predicted_note)
        
        # Append predicted index to sequence and shift window
        if use_embedding:
            current_sequence = np.append(current_sequence[1:], idx)
        else:
            # Normalized representation of the predicted note index
            normalized_idx = idx / float(vocab_size)
            current_sequence = np.append(current_sequence[1:], normalized_idx)
            
    print("Generation complete. Reconstructing MIDI stream...")
    
    # Reconstruct MIDI file using music21
    offset = 0.0
    output_notes = []
    
    for pattern in prediction_output:
        # If it's a chord (e.g. "C4.E4.G4")
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # If it's a single note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            
        # Increment offset to prevent notes overlapping sequentially
        # Adjusting the step size controls the speed of the output music
        offset += 0.5
        
    # Build stream
    midi_stream = stream.Stream(output_notes)
    
    # Save output files
    os.makedirs("output", exist_ok=True)
    midi_path = os.path.join("output", f"{output_name}.mid")
    midi_stream.write('midi', fp=midi_path)
    print(f"MIDI file saved to '{midi_path}'.")
    
    # Convert to WAV using our pure Python synthesizer
    wav_path = os.path.join("output", f"{output_name}.wav")
    try:
        synthesize_midi_to_wav(midi_path, wav_path)
    except Exception as e:
        print(f"Warning: Could not synthesize WAV audio ({e}).")
        print("Note: Install pretty_midi and scipy to enable WAV synthesis.")

def synthesize_midi_to_wav(midi_path, wav_path):
    """
    Lightweight, pure-Python synthesizer that reads a MIDI file via pretty_midi
    and synthesizes a playable WAV file using sine wave oscillators and ADSR envelopes.
    Avoids system-level fluidsynth binary dependencies.
    """
    import pretty_midi
    from scipy.io import wavfile
    
    print("Synthesizing audio wav file from MIDI...")
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    sr = 44100  # 44.1 kHz sampling rate
    duration = pm.get_end_time() + 1.0  # Pad with 1s tail
    num_samples = int(duration * sr)
    audio = np.zeros(num_samples)
    
    for instr in pm.instruments:
        for n in instr.notes:
            start_sample = int(n.start * sr)
            end_sample = int(n.end * sr)
            note_len = end_sample - start_sample
            if note_len <= 0:
                continue
                
            freq = pretty_midi.note_number_to_hz(n.pitch)
            t = np.linspace(0, n.end - n.start, note_len, endpoint=False)
            
            # Form base sine wave
            wave = np.sin(2 * np.pi * freq * t)
            
            # Add simple harmonics for a richer sound (mimicking piano timbre)
            wave += 0.4 * np.sin(2 * 2 * np.pi * freq * t)
            wave += 0.2 * np.sin(3 * 2 * np.pi * freq * t)
            
            # Apply ADSR envelope to remove clicks
            envelope = np.ones_like(t)
            # Attack: 15ms linear fade-in
            attack_len = min(int(0.015 * sr), note_len // 10)
            if attack_len > 0:
                envelope[:attack_len] = np.linspace(0, 1, attack_len)
            # Decay/Release: 50ms exponential-like fade-out
            decay_len = min(int(0.05 * sr), note_len // 5)
            if decay_len > 0:
                envelope[-decay_len:] = np.linspace(1, 0, decay_len)
                
            wave = wave * envelope * (n.velocity / 127.0) * 0.15
            
            # Mix note into master buffer
            if start_sample + len(wave) <= len(audio):
                audio[start_sample:start_sample+len(wave)] += wave
                
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8
        
    # Save as 16-bit PCM WAV file
    wavfile.write(wav_path, sr, (audio * 32767).astype(np.int16))
    print(f"Playable audio saved to '{wav_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music from trained LSTM model")
    parser.add_argument('--model', type=str, default="models/best_model.h5", help="Path to trained model file")
    parser.add_argument('--output', type=str, default="generated_music", help="Base name of output files")
    parser.add_argument('--notes', type=int, default=100, help="Number of notes to generate")
    parser.add_argument('--temp', type=float, default=0.7, help="Sampling temperature (creativity slider)")
    parser.add_argument('--no_embedding', action='store_true', help="Disable embedding layer")
    
    args = parser.parse_args()
    
    generate_music(
        model_path=args.model,
        output_name=args.output,
        num_notes=args.notes,
        temperature=args.temp,
        use_embedding=not args.no_embedding
    )
