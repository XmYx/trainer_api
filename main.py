from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import uuid
import shutil
import requests
import time
import subprocess
import torch
import gc
from glob import glob
import traceback
from PIL import Image
from tqdm import tqdm

# Import utilities required by flux_train
from utilities import (
    delete_data,
    download_zip,
    format_flux_data,
    upload_files_to_s3,
    upload_files_to_s3_images,
    get_vram_info,
    update_config,
)

# Assuming 'caption_generator' is a module for generating captions
from caption_generator import caption_flan, unload_flan_model

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
    name: str = Form(...),
    trigger_word: str = Form(...),
    instance_prompt: str = Form(...),
    theme: str = Form(...),
    steps: int = Form(500),
    auto_caption: bool = Form(True),
    content_or_style: str = Form('balanced'),
    batch_size: int = Form(2),
    gradient_accumulation_steps: int = Form(2)
):
    # Generate a unique ID for this training session
    request_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save the uploaded zip file
    zip_path = os.path.join(session_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(await zip_file.read())

    # Prepare input_json for flux_train
    input_json = {
        "request_id": request_id,
        "data_source_path": zip_path,
        "name": name,
        "trigger_word": trigger_word,
        "instance_prompt": instance_prompt,
        "theme": theme,
        "advance_parameters": {
            "steps": steps,
            "auto_caption": auto_caption,
            "content_or_style": content_or_style,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps
        }
    }

    # Start the training in the background
    background_tasks.add_task(
        flux_train,
        input_json=input_json,
        session_dir=session_dir
    )

    # Return the request ID to the user
    return {"status": "Training started", "request_id": request_id}

# Endpoint to check training status
@app.get("/status/{request_id}")
def get_status(request_id: str):
    session_dir = os.path.join(UPLOAD_DIR, request_id)
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
@app.get("/download/{request_id}")
def download_model(request_id: str):
    session_dir = os.path.join(UPLOAD_DIR, request_id)
    model_path = os.path.join(session_dir, "final_model", f"{request_id}.safetensors")
    if os.path.exists(model_path):
        return FileResponse(
            model_path,
            media_type='application/octet-stream',
            filename=f"{request_id}.safetensors"
        )
    else:
        return JSONResponse(status_code=404, content={"error": "Model not found"})

def flux_train(input_json, session_dir):
    try:
        # Read the json parameters
        body = input_json
        request_id = body["request_id"]
        # Mandatory parameters
        url = body.get("data_source_path", "")
        name = body.get("name")
        trigger_word = body.get("trigger_word")
        prompts = body.get("instance_prompt")
        theme = body.get("theme").lower()
        # Advanced parameters
        params = body.get("advance_parameters", {})
        print('params:', params)
        steps = params.get("steps", 500)
        auto_caption = params.get("auto_caption", True)
        content_or_style = str(params.get("content_or_style", 'balanced'))
        batch_size = params.get("batch_size", 2)
        gradient_accumulation_steps = params.get("gradient_accumulation_steps", 2)
        # Prepare the data
        # You may need to adjust the status update logic
        # For now, we'll just print the status
        print('Status: DATA_PROCESSING')
        delete_data(os.path.join(session_dir, "inputs", "images"))
        delete_data(os.path.join(session_dir, "inputs", "temp"))
        delete_data(os.path.join(session_dir, "outputs"))
        temp_yaml = os.path.join(session_dir, "flux_lora_config_temp.yaml")
        if os.path.isfile(temp_yaml):
            os.remove(temp_yaml)
        if url != "":
            # Assume download_zip downloads the zip file to session_dir
            # and unzips it into session_dir/inputs/images
            download_zip(url, os.path.join(session_dir, "inputs"))
            format_flux_data(url, os.path.join(session_dir, "inputs"))
            train_data_dir = os.path.join(session_dir, "inputs", "images")
            try:
                dataset_size = len(glob(train_data_dir + "/*"))
            except:
                dataset_size = 0
            try:
                file_size = os.path.getsize(url)
            except:
                file_size = 0

        before_load = 0
        st = time.time()
        if auto_caption:
            print('Generating captions')
            imgs = glob(f"{train_data_dir}/*")
            for img_path in tqdm(imgs):
                ext = img_path.split('/')[-1].split('.')[-1]
                if ext.lower() in ['jpg', 'png', 'jpeg']:
                    try:
                        img = Image.open(img_path)
                        caption = caption_flan(img)
                        with open(img_path.replace(f'.{ext}', '.txt'), 'w') as out:
                            out.write(caption)
                    except Exception as e:
                        print(f"Failed to generate caption for {img_path}: {e}")
                        print("Traceback:", str(traceback.format_exc()))
            unload_flan_model()

        if steps > 4000:
            steps = 4000
        if batch_size > 4:
            batch_size = 4
        if gradient_accumulation_steps > 4:
            gradient_accumulation_steps = 4

        params['trigger_word'] = trigger_word
        params['name'] = name
        params['prompts'] = [prompts + ' ' + trigger_word]
        params['content_or_style'] = content_or_style
        params['batch_size'] = batch_size
        params['gradient_accumulation_steps'] = gradient_accumulation_steps
        params['steps'] = steps
        # UPDATE CONFIG
        config_path = os.path.join(session_dir, "flux_lora_config_temp.yaml")
        update_config(params, config_path, train_data_dir)
        # Set the params
        command = [
            "python3",
            "ai-toolkit/run.py",
            config_path
        ]
        print('Status: TRAINING_INITIATED')
        # before_load = get_vram_info()
        subprocess.run(command)
        model_file = os.path.join(os.getcwd(), f'outputs/{name}/{name}.safetensors')
        if not os.path.exists(model_file):
            print('Model file does not exist in the path')
            # Update status
            with open(os.path.join(session_dir, "status.txt"), "w") as f:
                f.write("Training failed")
            raise Exception('Training failed')

        # Move weights
        final_model_dir = os.path.join(session_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        shutil.move(
            model_file,
            os.path.join(final_model_dir, f"{request_id}.safetensors")
        )
        # Move images
        test_images_dir = os.path.join(session_dir, "test_images")
        os.makedirs(test_images_dir, exist_ok=True)
        for i, image in enumerate(glob(os.path.join(os.getcwd(), f'outputs/{name}/samples/*'))):
            print(image)
            shutil.move(image, os.path.join(test_images_dir, f'sample_{i+1}.png'))

        # Upload files to S3 or any cloud storage if needed
        # For now, we'll skip uploading
        gc.collect()
        torch.cuda.empty_cache()

        et = time.time()
        train_time = et - st
        # after_load = get_vram_info()
        # vram = after_load - before_load
        # training_metadata = {"train_time": train_time, "steps": steps, "file_size": file_size}
        # model_metadata = {"vram": vram, "dataset_size": dataset_size, "file_size": file_size}
        # Update status
        with open(os.path.join(session_dir, "status.txt"), "w") as f:
            f.write("Training completed")
    except Exception as e:
        # Update status
        with open(os.path.join(session_dir, "status.txt"), "w") as f:
            f.write(f"Training failed: {str(e)}")
        print(f"Error during training: {e}")
        print("Traceback:", str(traceback.format_exc()))

# You may need to adjust the implementations of the following functions

def update_config(params, config_path, dataset_path):
    # Load the template config file
    with open("config_template.yml", 'r') as file:
        config = yaml.safe_load(file)

    # Update config with parameters
    config['config']['name'] = params['name']
    process = config['config']['process'][0]
    process['training_folder'] = os.path.join("outputs", params['name'])
    process['datasets'][0]['folder_path'] = dataset_path
    process['datasets'][0]['resolution'] = [512, 768, 1024]
    process['train']['lr'] = params.get('lr', 1e-4)
    process['train']['steps'] = params['steps']
    process['train']['batch_size'] = params['batch_size']
    process['train']['gradient_accumulation_steps'] = params['gradient_accumulation_steps']
    process['sample']['prompts'] = params['prompts']
    # Save the updated config
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

# Implement other utility functions or import them from 'utilities'

if __name__ == "__main__":
    # Run the app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
