import os
import urllib.request
import zipfile
import pickle
import numpy as np
from music21 import converter, note, chord, stream, meter, tempo, key, instrument

def download_midi_dataset(dest_dir="data/midi"):
    """
    Downloads a small subset of classical piano MIDI files (Mozart) from a public domain resource.
    If the download fails (e.g. offline), it falls back to programmatically generating synthetic midi files
    to ensure the pipeline runs robustly.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = "http://www.piano-midi.de/zip/mozart.zip"
    zip_path = os.path.join(dest_dir, "mozart.zip")
    
    print("--- STEP 1: Data Collection ---")
    try:
        print(f"Attempting to download classical MIDI dataset from {url}...")
        # Use a user-agent to avoid HTTP 403 Forbidden errors
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print("Download complete. Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
            
        # Clean up zip file
        os.remove(zip_path)
        print("Extraction complete. Classical MIDI dataset ready.")
        
    except Exception as e:
        print(f"Warning: Could not download MIDI dataset ({e}).")
        print("Falling back to programmatically generating a synthetic MIDI dataset for training...")
        create_synthetic_midi(dest_dir, num_files=20)
        print("Synthetic MIDI dataset generated successfully.")

def create_synthetic_midi(output_dir, num_files=25):
    """
    Generates rich, expressive classical piano MIDI files featuring famous classical
    progressions (Pachelbel Canon, Mozart Allegro, Bach Prelude, Chopin Nocturne, Beethoven motifs).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    classical_themes = [
        # Theme 1: Pachelbel Canon Harmony (D Major / C Major transpose)
        {
            'key': 'C', 'tempo': 84,
            'chords': [['C4','E4','G4'], ['G3','B3','D4'], ['A3','C4','E4'], ['E3','G3','B3'], ['F3','A3','C4'], ['C3','E3','G3'], ['F3','A3','C4'], ['G3','B3','D4']],
            'melody': ['C5','B4','A4','G4','F4','E4','F4','D4','E4','G4','C5','E5','D5','C5','B4','C5']
        },
        # Theme 2: Mozart K.545 Sonata Allegro (Bright & Joyful)
        {
            'key': 'C', 'tempo': 124,
            'chords': [['C4','E4','G4'], ['G3','B3','D4','F4'], ['C4','E4','G4'], ['F3','A3','C4'], ['G3','B3','D4'], ['C4','E4','G4']],
            'melody': ['C5','E5','G5','B4','C5','D5','C5','G4','E4','C4','F4','A4','C5','B4','A4','G4']
        },
        # Theme 3: Für Elise Motifs (Emotional Classical Minor)
        {
            'key': 'A', 'tempo': 96,
            'chords': [['A3','C4','E4'], ['E3','G#3','B3'], ['A3','C4','E4'], ['C4','E4','G4'], ['G3','B3','D4'], ['A3','C4','E4']],
            'melody': ['E5','D#5','E5','D#5','E5','B4','D5','C5','A4','C4','E4','A4','B4','E4','G#4','B4','C5']
        },
        # Theme 4: Chopin Nocturne Warm & Romantic
        {
            'key': 'G', 'tempo': 76,
            'chords': [['G3','B3','D4'], ['E3','G3','B3'], ['C4','E4','G4'], ['D3','F#3','A3','C4']],
            'melody': ['D5','B5','A5','G5','E5','D5','C5','B4','A4','B4','D5','G5','F#5','E5','D5']
        },
        # Theme 5: Bach Harmonic Prelude Arpeggios
        {
            'key': 'C', 'tempo': 92,
            'chords': [['C3','E4','G4','C5'], ['C3','D4','F4','A4'], ['B2','D4','G4','D5'], ['C3','E4','G4','C5']],
            'melody': ['C4','E4','G4','C5','E5','G4','C5','E5','C4','D4','F4','A4','D5','F4','A4','D5']
        }
    ]

    for i in range(num_files):
        theme = classical_themes[i % len(classical_themes)]
        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=theme['tempo']))
        s.append(meter.TimeSignature('4/4'))
        s.append(key.Key(theme['key']))
        
        offset = 0.0
        chords = theme['chords']
        melody = theme['melody']
        
        # Build 48 musical steps per song (chords + melodic embellishments)
        for step in range(48):
            if step % 4 == 0:
                c_notes = chords[(step // 4) % len(chords)]
                c = chord.Chord(c_notes)
                c.duration.quarterLength = 1.0
                c.offset = offset
                s.append(c)
                offset += 1.0
            else:
                m_note = melody[step % len(melody)]
                n = note.Note(m_note)
                n.duration.quarterLength = 0.5
                n.offset = offset
                s.append(n)
                offset += 0.5
                
        p = instrument.Piano()
        s.insert(0, p)
        
        file_path = os.path.join(output_dir, f"synthetic_song_{i}.mid")
        s.write('midi', fp=file_path)

def parse_midi_files(data_dir):
    """
    Parses all MIDI files in a directory using music21.
    Extracts note/chord sequences, along with their duration and offset metadata.
    """
    print("\n--- STEP 2: Preprocessing ---")
    notes = []
    durations = []
    offsets = []
    
    midi_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.mid') or f.endswith('.midi')
    ]
    
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}")
        
    print(f"Found {len(midi_files)} MIDI files to parse.")
    
    for idx, file_path in enumerate(midi_files):
        print(f"Parsing file {idx+1}/{len(midi_files)}: {os.path.basename(file_path)}")
        try:
            # Parse MIDI file
            midi = converter.parse(file_path)
            
            # Flatten to extract notes and chords
            # Compatible with both music21 v10+ (flatten()) and older versions (.flat)
            if hasattr(midi, 'flatten'):
                notes_to_parse = midi.flatten().notes
            else:
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                # Get offset relative to the start of the score
                offset_val = float(element.offset)
                duration_val = float(element.duration.quarterLength)
                
                if isinstance(element, note.Note):
                    pitch_str = str(element.pitch)
                    notes.append(pitch_str)
                    durations.append(duration_val)
                    offsets.append(offset_val)
                elif isinstance(element, chord.Chord):
                    # Join note pitches with a dot, e.g. "C4.E4.G4"
                    pitch_str = '.'.join(str(p) for p in element.pitches)
                    notes.append(pitch_str)
                    durations.append(duration_val)
                    offsets.append(offset_val)
                    
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            
    print(f"Parsed total of {len(notes)} note/chord elements.")
    return notes, durations, offsets

def prepare_sequences(notes, sequence_length=50):
    """
    Creates input sequences and output labels mapping unique note/chords to integers.
    Normalizes/reshapes for LSTM input.
    """
    # Get all unique notes/chords
    pitches = sorted(list(set(notes)))
    vocab_size = len(pitches)
    print(f"Vocabulary size (unique notes/chords): {vocab_size}")
    
    # Mapping unique notes to integers
    note_to_int = {note: number for number, note in enumerate(pitches)}
    int_to_note = {number: note for number, note in enumerate(pitches)}
    
    network_input = []
    network_output = []
    
    # Generate sequence pairs
    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])
        
    n_patterns = len(network_input)
    print(f"Total training sequences generated: {n_patterns}")
    
    # Convert to numpy arrays
    X = np.reshape(network_input, (n_patterns, sequence_length))
    
    # Reshape and normalize input for the case where we don't use embedding
    # (samples, timesteps, features = 1)
    X_normalized = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X_normalized = X_normalized / float(vocab_size)
    
    # One-hot encode outputs
    try:
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(network_output, num_classes=vocab_size)
    except ModuleNotFoundError:
        y = np.eye(vocab_size)[network_output]
    
    return X, X_normalized, y, note_to_int, int_to_note, vocab_size

def run_preprocessing(sequence_length=50):
    """
    Executes the preprocessing pipeline: downloads dataset, parses MIDI,
    creates numerical vocab mappings, and saves prepared structures.
    """
    data_dir = "data/midi"
    download_midi_dataset(data_dir)
    
    notes, durations, offsets = parse_midi_files(data_dir)
    
    # Save raw notes and metadata for analysis/fallback
    os.makedirs("data", exist_ok=True)
    with open("data/raw_notes.pkl", "wb") as f:
        pickle.dump((notes, durations, offsets), f)
        
    X, X_normalized, y, note_to_int, int_to_note, vocab_size = prepare_sequences(notes, sequence_length)
    
    # Save variables for model training
    with open("data/preprocessed_data.pkl", "wb") as f:
        pickle.dump({
            'X': X,
            'X_normalized': X_normalized,
            'y': y,
            'note_to_int': note_to_int,
            'int_to_note': int_to_note,
            'vocab_size': vocab_size,
            'sequence_length': sequence_length
        }, f)
        
    print("Preprocessing completed. Structured datasets saved to 'data/preprocessed_data.pkl'.")

if __name__ == "__main__":
    run_preprocessing()
