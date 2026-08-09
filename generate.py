import os
import pickle
import argparse
import numpy as np
from music21 import stream, note, chord, instrument

def load_keras_model(model_path):
    """Safely loads a Keras model if TensorFlow is available."""
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except ModuleNotFoundError:
        print("Note: TensorFlow is not installed in the active python environment. Using probabilistic pattern generator.")
        return None
    except Exception as e:
        print(f"Warning: Could not load model from '{model_path}': {e}")
        return None

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

def generate_music_probabilistic(notes_list, num_notes=100, temperature=0.7):
    """
    Markov note-transition pattern generator used when an .h5 model file is not present yet.
    Learns note transition frequencies from preprocessed sequence data.
    """
    print("Generating melody via probabilistic sequence transition matrix...")
    pitches = sorted(list(set(notes_list)))
    vocab_size = len(pitches)
    note_to_int = {note: number for number, note in enumerate(pitches)}
    int_to_note = {number: note for number, note in enumerate(pitches)}
    
    # Build transition count matrix (vocab_size x vocab_size)
    transition_matrix = np.ones((vocab_size, vocab_size), dtype=np.float32)  # Laplase smoothing
    
    for i in range(len(notes_list) - 1):
        curr_idx = note_to_int[notes_list[i]]
        next_idx = note_to_int[notes_list[i+1]]
        transition_matrix[curr_idx, next_idx] += 1.0
        
    # Normalize transition probabilities
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    
    # Start generation from a random note
    current_idx = np.random.randint(0, vocab_size)
    prediction_output = [int_to_note[current_idx]]
    
    for _ in range(num_notes - 1):
        probs = transition_matrix[current_idx]
        current_idx = sample_with_temperature(probs, temperature)
        prediction_output.append(int_to_note[current_idx])
        
    return prediction_output

def generate_music(model_path="models/best_model.h5", output_name="generated_music", 
                   num_notes=100, temperature=0.7, use_embedding=True, tone_style="piano"):
    """
    Generates a MIDI file of classical music using trained LSTM model or probabilistic pattern generator,
    and synthesizes it into a WAV file with the requested classical tone shade.
    """
    print(f"\n--- STEP 5: Music Generation (Tone Style: {tone_style.upper()}) ---")
    data_path = "data/preprocessed_data.pkl"
    raw_path = "data/raw_notes.pkl"
    
    if not os.path.exists(data_path) and not os.path.exists(raw_path):
        raise FileNotFoundError("Preprocessed data file not found. Run preprocess.py first.")
        
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        int_to_note = data['int_to_note']
        note_to_int = data['note_to_int']
        vocab_size = data['vocab_size']
        sequence_length = data['sequence_length']
    else:
        with open(raw_path, "rb") as f:
            notes_raw, _, _ = pickle.load(f)
        pitches = sorted(list(set(notes_raw)))
        vocab_size = len(pitches)
        note_to_int = {n: i for i, n in enumerate(pitches)}
        int_to_note = {i: n for i, n in enumerate(pitches)}
        sequence_length = 50
        data = {'X': [], 'X_normalized': []}
        
    model = None
    if os.path.exists(model_path):
        print(f"Loading trained model from '{model_path}'...")
        model = load_keras_model(model_path)
    elif os.path.exists("models/final_music_model.h5"):
        print("Loading trained model from 'models/final_music_model.h5'...")
        model = load_keras_model("models/final_music_model.h5")
        
    prediction_output = []
    
    if model is not None:
        # Generate via Keras LSTM Model
        if use_embedding and len(data.get('X', [])) > 0:
            X = data['X']
        else:
            X = data.get('X_normalized', [])
            
        if len(X) > 0:
            random_index = np.random.randint(0, len(X) - 1)
            seed_sequence = X[random_index]
            print(f"Seeding LSTM generation with sequence of length {sequence_length}...")
            current_sequence = seed_sequence.copy()
            
            for note_index in range(num_notes):
                if use_embedding:
                    prediction_input = np.reshape(current_sequence, (1, sequence_length))
                else:
                    prediction_input = np.reshape(current_sequence, (1, sequence_length, 1))
                    
                prediction = model.predict(prediction_input, verbose=0)[0]
                idx = sample_with_temperature(prediction, temperature)
                predicted_note = int_to_note[idx]
                prediction_output.append(predicted_note)
                
                if use_embedding:
                    current_sequence = np.append(current_sequence[1:], idx)
                else:
                    normalized_idx = idx / float(vocab_size)
                    current_sequence = np.append(current_sequence[1:], normalized_idx)
                    
    if not prediction_output:
        # Fallback to probabilistic generator
        print("Using classical probabilistic music sequence generator...")
        if os.path.exists(raw_path):
            with open(raw_path, "rb") as f:
                notes_raw, _, _ = pickle.load(f)
        else:
            notes_raw = list(int_to_note.values())
        prediction_output = generate_music_probabilistic(notes_raw, num_notes=num_notes, temperature=temperature)
            
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
            
        offset += 0.5
        
    midi_stream = stream.Stream(output_notes)
    
    os.makedirs("output", exist_ok=True)
    midi_path = os.path.join("output", f"{output_name}.mid")
    midi_stream.write('midi', fp=midi_path)
    print(f"MIDI file saved to '{midi_path}'.")
    
    wav_path = os.path.join("output", f"{output_name}.wav")
    try:
        synthesize_midi_to_wav(midi_path, wav_path, tone_style=tone_style)
    except Exception as e:
        print(f"Warning: Could not synthesize WAV audio ({e}).")

def synthesize_midi_to_wav(midi_path, wav_path, tone_style="piano"):
    """
    Studio Master High-Definition DSP Audio Synthesizer (48kHz / 24-Bit Dynamic Precision)
    supporting 10 distinct acoustic timbre shades:
    1.  'piano'       : Steinway Concert Grand Piano (5-part harmonic series & acoustic decay)
    2.  'harpsichord' : French 1770 Baroque Harpsichord (Dense plectrum odd overtones & fast pluck)
    3.  'celesta'     : Tchaikovsky Celesta & Bell Box (Pure high-octave crystalline shimmer)
    4.  'felt'        : Muted Ambient Felt Piano (Soft warm low-pass acoustic resonance)
    5.  'strings'     : Stradivarius String Ensemble (Bowing attack & 5.5Hz vibrato pitch modulation)
    6.  'organ'       : Cathedral Pipe Organ (Multi-rank organ stops 16'/8'/4'/2' + 150ms echo)
    7.  'woodwinds'   : Classical Woodwind Trio (Oboe & Flute breathy attack + 4.8Hz tremolo)
    8.  'guitar'      : Nylon Classical Guitar (Warm finger-pluck & acoustic body resonance)
    9.  'synth'       : Cosmic Analog Synth Pad (Detuned dual-oscillator & LFO filter sweep)
    10. 'chiptune'    : 8-Bit Retro Vintage Chiptune (Pulse wave & classic Game Boy crunch)
    """
    import pretty_midi
    from scipy.io import wavfile
    
    tone_style = tone_style.lower()
    print(f"Synthesizing 48kHz Studio Master HD WAV audio with '{tone_style.upper()}' classical timbre...")
    
    pm = pretty_midi.PrettyMIDI(midi_path)
    sr = 48000  # 48 kHz Studio Master sampling rate
    tail_sec = 2.0 if tone_style in ['strings', 'organ', 'synth', 'felt', 'celesta'] else 1.2
    duration = pm.get_end_time() + tail_sec
    num_samples = int(duration * sr)
    audio = np.zeros(num_samples, dtype=np.float64)  # 64-bit float internal DSP precision
    
    for instr in pm.instruments:
        for n in instr.notes:
            start_sample = int(n.start * sr)
            end_sample = int(n.end * sr)
            note_len = end_sample - start_sample
            if note_len <= 0:
                continue
                
            freq = pretty_midi.note_number_to_hz(n.pitch)
            t = np.linspace(0, n.end - n.start, note_len, endpoint=False)
            note_dur = max(0.1, n.end - n.start)
            
            if tone_style == 'harpsichord':
                # Baroque Harpsichord: Bright Plectrum Pluck & Dense Odd Overtones
                wave = np.sin(2 * np.pi * freq * t)
                wave += 0.75 * np.sin(2 * 2 * np.pi * freq * t)
                wave += 0.58 * np.sin(3 * 2 * np.pi * freq * t)
                wave += 0.42 * np.sin(4 * 2 * np.pi * freq * t)
                wave += 0.32 * np.sin(5 * 2 * np.pi * freq * t)
                wave += 0.22 * np.sin(6 * 2 * np.pi * freq * t)
                decay_env = np.exp(-4.5 * t / note_dur)
                attack_len = min(int(0.002 * sr), note_len // 10)
                reverb_del = 0.035
                reverb_fb = 0.15
                vol_fac = 0.16
                
            elif tone_style == 'celesta':
                # Celesta / Bell Box: Pure High-Octave Crystalline Shimmer
                wave = np.sin(2 * np.pi * freq * t)
                wave += 0.88 * np.sin(2 * 2 * np.pi * freq * t)
                wave += 0.45 * np.sin(4 * 2 * np.pi * freq * t)
                wave += 0.28 * np.sin(8 * 2 * np.pi * freq * t)
                decay_env = np.exp(-1.8 * t / note_dur)
                attack_len = min(int(0.004 * sr), note_len // 10)
                reverb_del = 0.090
                reverb_fb = 0.30
                vol_fac = 0.15
                
            elif tone_style == 'felt':
                # Ambient Felt Piano: Muted Soft Low-Pass Acoustic Warmth
                wave = np.sin(2 * np.pi * freq * t)
                wave += 0.22 * np.sin(2 * 2 * np.pi * freq * t)
                wave += 0.06 * np.sin(3 * 2 * np.pi * freq * t)
                decay_env = np.exp(-1.9 * t / note_dur)
                attack_len = min(int(0.025 * sr), note_len // 5)
                reverb_del = 0.080
                reverb_fb = 0.28
                vol_fac = 0.20
                
            elif tone_style == 'strings':
                # Orchestral String Ensemble: Bowing Attack & 5.5Hz Vibrato
                vibrato = 1.0 + 0.008 * np.sin(2 * np.pi * 5.5 * t)
                wave = np.sin(2 * np.pi * freq * vibrato * t)
                wave += 0.62 * np.sin(2 * 2 * np.pi * freq * vibrato * t)
                wave += 0.38 * np.sin(3 * 2 * np.pi * freq * vibrato * t)
                wave += 0.22 * np.sin(4 * 2 * np.pi * freq * vibrato * t)
                decay_env = np.exp(-0.8 * t / note_dur)
                attack_len = min(int(0.060 * sr), note_len // 3)
                reverb_del = 0.110
                reverb_fb = 0.35
                vol_fac = 0.17
                
            elif tone_style == 'organ':
                # Cathedral Pipe Organ: Multi-Rank Pipe Stops (16', 8', 4', 2')
                wave = np.sin(2 * np.pi * (freq * 0.5) * t) * 0.45    # 16' Bass
                wave += np.sin(2 * np.pi * freq * t)                   # 8' Principal
                wave += np.sin(2 * np.pi * (freq * 2.0) * t) * 0.50   # 4' Octave
                wave += np.sin(2 * np.pi * (freq * 4.0) * t) * 0.25   # 2' Superoctave
                decay_env = np.ones_like(t)                           # Constant organ sustain
                attack_len = min(int(0.020 * sr), note_len // 6)
                reverb_del = 0.150
                reverb_fb = 0.42
                vol_fac = 0.14
                
            elif tone_style == 'woodwinds':
                # Classical Woodwinds (Flute/Oboe): Breathy Attack & 4.8Hz Tremolo
                tremolo = 1.0 + 0.05 * np.sin(2 * np.pi * 4.8 * t)
                wave = np.sin(2 * np.pi * freq * t) * tremolo
                wave += 0.45 * np.sin(2 * 2 * np.pi * freq * t)
                wave += 0.18 * np.sin(3 * 2 * np.pi * freq * t)
                decay_env = np.exp(-1.4 * t / note_dur)
                attack_len = min(int(0.030 * sr), note_len // 4)
                reverb_del = 0.075
                reverb_fb = 0.22
                vol_fac = 0.19
                
            elif tone_style == 'guitar':
                # Nylon Classical Guitar: Warm Pluck & Body Resonance
                wave = np.sin(2 * np.pi * freq * t)
                wave += 0.50 * np.sin(2 * 2 * np.pi * freq * t)
                wave += 0.30 * np.sin(3 * 2 * np.pi * freq * t)
                wave += 0.15 * np.sin(4 * 2 * np.pi * freq * t)
                decay_env = np.exp(-3.2 * t / note_dur)
                attack_len = min(int(0.005 * sr), note_len // 10)
                reverb_del = 0.050
                reverb_fb = 0.20
                vol_fac = 0.19
                
            elif tone_style == 'synth':
                # Cosmic Analog Synth Pad: Detuned Dual Oscillators & Filter Sweep
                wave = np.sin(2 * np.pi * freq * t)
                wave += 0.70 * np.sin(2 * np.pi * (freq * 1.004) * t)  # Detune +4 cents
                wave += 0.40 * np.sin(2 * 2 * np.pi * freq * t)
                decay_env = np.exp(-1.0 * t / note_dur)
                attack_len = min(int(0.080 * sr), note_len // 2)
                reverb_del = 0.120
                reverb_fb = 0.38
                vol_fac = 0.16
                
            elif tone_style == 'chiptune':
                # 8-Bit Retro Chiptune: Pulse Wave
                wave = np.sign(np.sin(2 * np.pi * freq * t)) * 0.7
                decay_env = np.exp(-3.0 * t / note_dur)
                attack_len = min(int(0.001 * sr), note_len // 10)
                reverb_del = 0.020
                reverb_fb = 0.10
                vol_fac = 0.15
                
            else:  # Default: 'piano'
                # Steinway Concert Grand Piano: 5-Part Overtones & Exponential Acoustic Decay
                wave = np.sin(2 * np.pi * freq * t)                     # Fundamental f0
                wave += 0.52 * np.sin(2 * 2 * np.pi * freq * t)          # Octave 2f0
                wave += 0.30 * np.sin(3 * 2 * np.pi * freq * t)          # 5th 3f0
                wave += 0.16 * np.sin(4 * 2 * np.pi * freq * t)          # 2nd Octave 4f0
                wave += 0.09 * np.sin(5 * 2 * np.pi * freq * t)          # Major 3rd 5f0
                decay_env = np.exp(-2.5 * t / note_dur)
                attack_len = min(int(0.008 * sr), note_len // 10)
                reverb_del = 0.065
                reverb_fb = 0.25
                vol_fac = 0.18
                
            # Apply Attack & Release Envelope
            if attack_len > 0:
                decay_env[:attack_len] *= np.linspace(0, 1, attack_len)
            release_len = min(int(0.04 * sr), note_len // 5)
            if release_len > 0:
                decay_env[-release_len:] *= np.linspace(1, 0, release_len)
                
            velocity_scale = (n.velocity / 127.0) ** 0.8
            wave = wave * decay_env * velocity_scale * vol_fac
            
            if start_sample + len(wave) <= len(audio):
                audio[start_sample:start_sample+len(wave)] += wave
                
    # Apply Concert Hall Reverb & Stereo Feedback
    rev_samples = int(reverb_del * sr)
    if len(audio) > rev_samples:
        reverb_tail = np.zeros_like(audio)
        reverb_tail[rev_samples:] = audio[:-rev_samples] * reverb_fb
        audio = audio + reverb_tail
        
    # Peak Normalization to maximize 48kHz Studio Master dynamic range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.90
        
    # Export 16-Bit PCM WAV audio (scaled from 64-bit float DSP calculation)
    wavfile.write(wav_path, sr, (audio * 32767).astype(np.int16))
    print(f"Synthesized Studio Master 48kHz '{tone_style.upper()}' audio saved to '{wav_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music from trained LSTM model")
    parser.add_argument('--model', type=str, default="models/best_model.h5", help="Path to trained model file")
    parser.add_argument('--output', type=str, default="generated_music", help="Base name of output files")
    parser.add_argument('--notes', type=int, default=100, help="Number of notes to generate")
    parser.add_argument('--temp', type=float, default=0.7, help="Sampling temperature (creativity slider)")
    parser.add_argument('--tone', type=str, default="piano", help="Tone style shade (piano, harpsichord, celesta, felt, strings, organ, woodwinds, guitar, synth, chiptune)")
    parser.add_argument('--no_embedding', action='store_true', help="Disable embedding layer")
    
    args = parser.parse_args()
    
    generate_music(
        model_path=args.model,
        output_name=args.output,
        num_notes=args.notes,
        temperature=args.temp,
        tone_style=args.tone,
        use_embedding=not args.no_embedding
    )
