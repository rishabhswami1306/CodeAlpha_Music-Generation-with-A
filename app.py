import os
import sys
import subprocess
import json
import shutil
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global process trackers
preprocess_process = None
train_process = None

def get_log_tail(filepath, lines_count=50):
    """Utility to get the last N lines of a log file."""
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return "".join(lines[-lines_count:])
    except Exception as e:
        return f"Error reading log: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns general status of the pipeline (data, model, output availability)."""
    has_raw_data = os.path.exists("data/raw_notes.pkl")
    has_preprocessed = os.path.exists("data/preprocessed_data.pkl")
    
    # Check for models
    models = []
    if os.path.exists("models"):
        models = [f for f in os.listdir("models") if f.endswith('.h5')]
        
    # Check for outputs
    outputs = []
    if os.path.exists("output"):
        outputs = [f for f in os.listdir("output") if f.endswith('.mid') or f.endswith('.wav')]
        
    # Preprocess status
    preprocess_running = preprocess_process is not None and preprocess_process.poll() is None
    
    # Train status
    train_running = train_process is not None and train_process.poll() is None
    
    # Read training progress from file
    training_progress = None
    if os.path.exists("data/training_status.json"):
        try:
            with open("data/training_status.json", "r") as f:
                training_progress = json.load(f)
        except Exception:
            pass
            
    return jsonify({
        "preprocess_running": preprocess_running,
        "train_running": train_running,
        "has_raw_data": has_raw_data,
        "has_preprocessed": has_preprocessed,
        "models": models,
        "outputs": outputs,
        "training_progress": training_progress
    })

@app.route('/api/preprocess', methods=['POST'])
def run_preprocess():
    """Triggers preprocess.py as a background subprocess."""
    global preprocess_process
    
    # Check if already running
    if preprocess_process is not None and preprocess_process.poll() is None:
        return jsonify({"status": "error", "message": "Preprocessing is already running."}), 400
        
    os.makedirs("data", exist_ok=True)
    
    # Clean previous status/logs
    if os.path.exists("data/preprocess.log"):
        try:
            os.remove("data/preprocess.log")
        except Exception:
            pass
            
    # Launch preprocess.py
    try:
        preprocess_process = subprocess.Popen(
            [sys.executable, "preprocess.py"],
            stdout=open("data/preprocess.log", "w", encoding='utf-8'),
            stderr=subprocess.STDOUT,
            text=True
        )
        return jsonify({"status": "success", "message": "Preprocessing started."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start preprocessing: {str(e)}"}), 500

@app.route('/api/preprocess/status', methods=['GET'])
def get_preprocess_status():
    """Gets the status and log tail of the preprocessing script."""
    running = preprocess_process is not None and preprocess_process.poll() is None
    exit_code = preprocess_process.poll() if preprocess_process else None
    
    logs = get_log_tail("data/preprocess.log")
    
    status_str = "idle"
    if running:
        status_str = "running"
    elif exit_code == 0:
        status_str = "completed"
    elif exit_code is not None:
        status_str = "failed"
        
    return jsonify({
        "status": status_str,
        "logs": logs,
        "exit_code": exit_code
    })

@app.route('/api/train', methods=['POST'])
def run_train():
    """Triggers train.py as a background subprocess with custom epochs/batch size."""
    global train_process
    
    if train_process is not None and train_process.poll() is None:
        return jsonify({"status": "error", "message": "Training is already running."}), 400
        
    # Get parameters
    data = request.json or {}
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 64)
    
    os.makedirs("data", exist_ok=True)
    
    # Clean logs and reset progress status file
    if os.path.exists("data/train.log"):
        try:
            os.remove("data/train.log")
        except Exception:
            pass
            
    try:
        with open("data/training_status.json", "w") as f:
            json.dump({
                "status": "starting",
                "epoch": 0,
                "total_epochs": epochs,
                "loss": 0.0,
                "accuracy": 0.0
            }, f)
    except Exception:
        pass
        
    # Launch train.py
    try:
        train_process = subprocess.Popen(
            [sys.executable, "train.py", "--epochs", str(epochs), "--batch_size", str(batch_size)],
            stdout=open("data/train.log", "w", encoding='utf-8'),
            stderr=subprocess.STDOUT,
            text=True
        )
        return jsonify({"status": "success", "message": "Training started."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to start training: {str(e)}"}), 500

@app.route('/api/train/status', methods=['GET'])
def get_train_status():
    """Gets training progress details from status file and stdout log tail."""
    running = train_process is not None and train_process.poll() is None
    exit_code = train_process.poll() if train_process else None
    
    progress = {}
    if os.path.exists("data/training_status.json"):
        try:
            with open("data/training_status.json", "r") as f:
                progress = json.load(f)
        except Exception:
            pass
            
    logs = get_log_tail("data/train.log")
    
    # Determine overall status
    if running:
        status_str = "training"
    elif exit_code == 0:
        status_str = "completed"
    elif exit_code is not None:
        status_str = "failed"
        # Update JSON status if failed
        if progress.get("status") == "training":
            progress["status"] = "failed"
            progress["error"] = "Subprocess exited with error."
    else:
        status_str = "idle"
        
    return jsonify({
        "status": status_str,
        "progress": progress,
        "logs": logs,
        "exit_code": exit_code
    })

@app.route('/api/train/stop', methods=['POST'])
def stop_train():
    """Force terminates the training process."""
    global train_process
    
    if train_process is None or train_process.poll() is not None:
        return jsonify({"status": "error", "message": "Training is not running."}), 400
        
    try:
        # Terminate process tree
        train_process.terminate()
        train_process.wait(timeout=3)
        status_msg = "Training terminated."
    except subprocess.TimeoutExpired:
        train_process.kill()
        status_msg = "Training force killed."
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to stop training: {str(e)}"}), 500
        
    # Write stopped status
    try:
        with open("data/training_status.json", "w") as f:
            json.dump({
                "status": "stopped",
                "error": "Training stopped by user."
            }, f)
    except Exception:
        pass
        
    return jsonify({"status": "success", "message": status_msg})

@app.route('/api/generate', methods=['POST'])
def run_generate():
    """Generates music (MIDI & WAV) using generate.py."""
    data = request.json or {}
    notes = data.get('notes', 100)
    temp = data.get('temp', 0.7)
    model = data.get('model', 'models/best_model.h5')
    
    # Check if preprocessed data exists
    if not os.path.exists("data/preprocessed_data.pkl"):
        return jsonify({"status": "error", "message": "Preprocessed data not found. Run preprocessing first."}), 400
        
    # Check if model exists
    if not os.path.exists(model):
        # Check if fallback model exists
        if os.path.exists("models/final_music_model.h5"):
            model = "models/final_music_model.h5"
        else:
            return jsonify({"status": "error", "message": "No trained model found. Please train a model first."}), 400
            
    # Run generate.py as a subprocess to capture stdout/stderr easily and run it inside the correct env
    try:
        proc = subprocess.Popen(
            [sys.executable, "generate.py", "--model", model, "--notes", str(notes), "--temp", str(temp)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(timeout=60)
        
        if proc.returncode == 0:
            return jsonify({
                "status": "success",
                "message": "Music generated successfully!",
                "stdout": stdout
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Music generation failed.",
                "stderr": stderr,
                "stdout": stdout
            }), 500
            
    except subprocess.TimeoutExpired as e:
        return jsonify({"status": "error", "message": "Generation timed out."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error running generation: {str(e)}"}), 500

@app.route('/api/play', methods=['GET'])
def play_audio():
    """Serves the generated WAV audio for inline browser playing."""
    wav_path = "output/generated_music.wav"
    if not os.path.exists(wav_path):
        return "Audio not found. Generate music first.", 404
    return send_file(wav_path, mimetype="audio/wav")

@app.route('/api/download/<file_format>', methods=['GET'])
def download_file(file_format):
    """Downloads the generated MIDI or WAV file."""
    if file_format == 'midi':
        path = "output/generated_music.mid"
        filename = "generated_music.mid"
    elif file_format == 'wav':
        path = "output/generated_music.wav"
        filename = "generated_music.wav"
    else:
        return "Invalid file format.", 400
        
    if not os.path.exists(path):
        return "Requested file not found. Generate music first.", 404
        
    return send_file(path, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    print("Starting Music Generator Web Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
