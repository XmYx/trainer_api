# Flux LoRA Training Server

This repository provides a FastAPI server that allows users to upload a dataset of images and captions, adjust training parameters (such as learning rate and number of steps), train a Flux LoRA model using the AI Toolkit by Ostris, and download the trained model in `.safetensors` format.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Server](#starting-the-server)
  - [API Endpoints](#api-endpoints)
    - [1. Train Model](#1-train-model)
    - [2. Check Training Status](#2-check-training-status)
    - [3. Download Trained Model](#3-download-trained-model)
  - [Example Usage with `curl`](#example-usage-with-curl)
- [Configuration](#configuration)
- [Notes](#notes)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Accepts ZIP file uploads containing images and captions.
- Allows users to specify training parameters: learning rate and number of steps.
- Trains a Flux LoRA model using the AI Toolkit by Ostris.
- Provides endpoints to check training status and download the trained model.
- Handles multiple training sessions using unique session IDs.

## Requirements

- **Hardware**:
  - NVIDIA GPU with at least 24GB VRAM (e.g., RTX A6000, A40)
  - CUDA-compatible drivers installed

- **Software**:
  - Python 3.10 or higher
  - Git
  - Python packages:
    - `fastapi`
    - `uvicorn`
    - `pyyaml`
    - `torch`
    - Other dependencies as specified in the AI Toolkit's `requirements.txt`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/flux-lora-training-server.git
   cd flux-lora-training-server
   ```

2. **Clone the AI Toolkit Repository**

   ```bash
   git clone https://github.com/ostris/ai-toolkit.git
   cd ai-toolkit
   git submodule update --init --recursive
   ```

3. **Create a Virtual Environment and Activate It**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

4. **Install Dependencies**

   ```bash
   # Install PyTorch first (ensure you have the correct version for your CUDA)
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

   # Install AI Toolkit requirements
   pip install -r requirements.txt

   # Install server dependencies
   pip install fastapi uvicorn pyyaml python-multipart

   # Navigate back to the root directory
   cd ..
   ```

5. **Set Up Hugging Face Authentication**

   - **Accept Model Access**:
     - Sign into Hugging Face and accept access to the FLUX.1-dev model [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).

   - **Create an Access Token**:
     - Go to your Hugging Face [settings](https://huggingface.co/settings/tokens) and create a new token with `read` permissions.

   - **Create a `.env` File**:
     - In the root directory, create a file named `.env` and add your token:

       ```
       HF_TOKEN=your_huggingface_read_token
       ```

## Usage

### Starting the Server

Run the server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

- The server will start on `http://0.0.0.0:8000`.

### API Endpoints

#### 1. **Train Model**

- **Endpoint**: `/train`
- **Method**: `POST`
- **Description**: Upload a dataset and start training.
- **Parameters**:
  - `zip_file` (form data, required): The ZIP file containing images and captions.
  - `learning_rate` (form data, required): The learning rate for training (e.g., `0.0001`).
  - `steps` (form data, required): The number of training steps (e.g., `2000`).
- **Response**:
  - `status`: Training initiation status.
  - `session_id`: A unique identifier for the training session.

#### 2. **Check Training Status**

- **Endpoint**: `/status/{session_id}`
- **Method**: `GET`
- **Description**: Get the current status of the training session.
- **Parameters**:
  - `session_id` (path parameter, required): The session ID returned from the `/train` endpoint.
- **Response**:
  - `status`: The current status of the training session.

#### 3. **Download Trained Model**

- **Endpoint**: `/download/{session_id}`
- **Method**: `GET`
- **Description**: Download the trained model in `.safetensors` format.
- **Parameters**:
  - `session_id` (path parameter, required): The session ID returned from the `/train` endpoint.
- **Response**:
  - Returns the `.safetensors` file if training is complete.

### Example Usage with `curl`

#### **1. Start Training**

```bash
curl -X POST "http://localhost:8000/train" \
  -F "zip_file=@/path/to/your/dataset.zip" \
  -F "learning_rate=0.0001" \
  -F "steps=2000"
```

- **Response**:

  ```json
  {
    "status": "Training started",
    "session_id": "your_session_id"
  }
  ```

#### **2. Check Training Status**

```bash
curl "http://localhost:8000/status/your_session_id"
```

- **Response**:

  ```json
  {
    "status": "Training in progress"
  }
  ```

#### **3. Download Trained Model**

```bash
curl -O "http://localhost:8000/download/your_session_id"
```

- This will download the file `your_session_id.safetensors`.

## Configuration

- **Configuration Template**:
  - The server uses a configuration template `config_template.yml` to generate a custom config for each training session.
  - This template is located in the same directory as `main.py`.

- **Adjusting Parameters**:
  - You can modify `config_template.yml` to change default settings or add additional parameters.
  - The server script updates the config dynamically based on user inputs.

- **Session Management**:
  - Each training session has a unique directory under `user_data`, storing datasets, outputs, and status files.
  - Session IDs are generated using UUIDs to ensure uniqueness.

## Notes

- **Dataset Preparation**:
  - **Images**: Should be in `.jpg`, `.jpeg`, or `.png` formats.
  - **Captions**: Each image should have a corresponding `.txt` file with the same name (e.g., `image1.jpg` and `image1.txt`).
  - **ZIP File**: Compress your dataset folder (containing images and captions) into a ZIP file before uploading.

- **Hardware Requirements**:
  - The server must run on a machine with an NVIDIA GPU that meets the VRAM requirements.
  - Ensure CUDA drivers and toolkit are properly installed.

- **Hugging Face Model Access**:
  - You must have accepted the license agreement for `black-forest-labs/FLUX.1-dev` on Hugging Face.
  - The Hugging Face token in `.env` must have `read` permissions.

- **Environment Variables**:
  - The `.env` file is used to store sensitive information like the Hugging Face token.
  - Make sure not to commit `.env` to version control.

- **Security Considerations**:
  - If deploying publicly, implement authentication mechanisms to prevent unauthorized access.
  - Validate and sanitize all user inputs.
  - Consider setting limits on uploaded file sizes and resource usage.

- **Error Handling**:
  - The server provides basic error handling.
  - You may enhance it by adding more detailed exceptions and logging.

- **Logs and Outputs**:
  - Training outputs and logs are stored in the session's `output` directory.
  - You can monitor training progress by checking these logs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **AI Toolkit by Ostris**: [https://github.com/ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- **FLUX Models**: [https://huggingface.co/black-forest-labs](https://huggingface.co/black-forest-labs)
- **FastAPI**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Uvicorn**: [https://www.uvicorn.org/](https://www.uvicorn.org/)

---

Feel free to contribute or open issues for any problems you encounter.
