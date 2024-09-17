from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import yaml
import subprocess
import zipfile
import os
import uuid
import shutil

app = FastAPI()

# Define a base path for all operations
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

# Define a directory to store user uploads and outputs
UPLOAD_DIR = os.path.join(BASE_PATH, "user_data")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Endpoint to accept training requests
@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(...),
    learning_rate: float = Form(...),
    steps: int = Form(...)
):
    # Generate a unique ID for this training session
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save the uploaded zip file
    zip_path = os.path.join(session_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(await zip_file.read())

    # Unzip the contents
    dataset_dir = os.path.join(session_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    # Update the config file
    config_path = os.path.join(session_dir, "config.yml")
    update_config(
        config_template_path=os.path.join(BASE_PATH, "config_template.yml"),
        output_config_path=config_path,
        dataset_path=dataset_dir,
        learning_rate=learning_rate,
        steps=steps,
        session_id=session_id
    )

    # Start the training in the background
    background_tasks.add_task(
        start_training,
        config_path=config_path,
        session_dir=session_dir
    )

    # Return the session ID to the user
    return {"status": "Training started", "session_id": session_id}

# Endpoint to check training status
@app.get("/status/{session_id}")
def get_status(session_id: str):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if not os.path.exists(session_dir):
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    status_file = os.path.join(session_dir, "status.txt")
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = f.read()
    else:
        status = "Training in progress"

    return {"status": status}

# Endpoint to download the trained model
@app.get("/download/{session_id}")
def download_model(session_id: str):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    model_path = os.path.join(session_dir, f"{session_id}.safetensors")
    if os.path.exists(model_path):
        return FileResponse(
            model_path,
            media_type='application/octet-stream',
            filename=f"{session_id}.safetensors"
        )
    else:
        return JSONResponse(status_code=404, content={"error": "Model not found"})

def update_config(config_template_path, output_config_path, dataset_path, learning_rate, steps, session_id):
    # Load the template config file
    with open(config_template_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update config with user parameters
    config['config']['name'] = session_id
    process = config['config']['process'][0]
    process['training_folder'] = os.path.join(session_dir, "output")
    process['datasets'][0]['folder_path'] = dataset_path
    process['train']['lr'] = learning_rate
    process['train']['steps'] = steps

    # Save the updated config
    with open(output_config_path, 'w') as file:
        yaml.dump(config, file)

def start_training(config_path, session_dir):
    status_file = os.path.join(session_dir, "status.txt")
    try:
        # Write initial status
        with open(status_file, "w") as f:
            f.write("Training in progress")

        # Run the training script
        subprocess.run(["python", "run.py", config_path], check=True)

        # After training, move the trained model to session directory
        output_dir = os.path.join(session_dir, "output")
        trained_model = find_safetensors_file(output_dir)
        if trained_model:
            shutil.move(trained_model, os.path.join(session_dir, f"{session_id}.safetensors"))
            # Update status
            with open(status_file, "w") as f:
                f.write("Training completed")
        else:
            # Update status
            with open(status_file, "w") as f:
                f.write("Training failed: Model not found")
    except subprocess.CalledProcessError as e:
        # Update status
        with open(status_file, "w") as f:
            f.write(f"Training failed: {str(e)}")

def find_safetensors_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".safetensors"):
                return os.path.join(root, file)
    return None

if __name__ == "__main__":
    # Run the app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
