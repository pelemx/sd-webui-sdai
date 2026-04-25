import os
import gc
import threading
from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import gradio as gr
from modules import script_callbacks
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Define extension paths
EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MODELS_DIR = os.path.join(EXTENSION_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# State Management for the loaded model
class LLMState:
    model: Llama = None
    model_name: str = None
    download_status: str = "Idle"

state = LLMState()

def download_model_task(repo_id: str, filename: str):
    """Background task to download Hugging Face GGUF models."""
    state.download_status = f"Downloading {filename} from {repo_id}..."
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        state.download_status = f"Success: Saved to {downloaded_path}"
    except Exception as e:
        state.download_status = f"Error: {str(e)}"

def mount_sdai_api(_: gr.Blocks, app: FastAPI):
    
    @app.get("/sdai/api/status")
    def get_status():
        return {
            "loaded_model": state.model_name,
            "download_task": state.download_status,
            "gpu_layers_active": state.model is not None
        }

    @app.get("/sdai/api/models")
    def list_models():
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.gguf')]
        return {"available_models": files}

    @app.post("/sdai/api/pullmodel")
    def pull_model(background_tasks: BackgroundTasks, repo_id: str = Body(...), filename: str = Body(...)):
        """Example body: {"repo_id": "Qwen/Qwen1.5-4B-Chat-GGUF", "filename": "qwen1_5-4b-chat-q4_k_m.gguf"}"""
        background_tasks.add_task(download_model_task, repo_id, filename)
        return {"status": "Download started in background. Check /sdai/api/status"}

    @app.post("/sdai/api/action")
    def manage_action(action: str = Body(...), target_model: str = Body(None)):
        """Actions: load, unload, delete"""
        if action == "unload":
            if state.model:
                del state.model
                state.model = None
                state.model_name = None
                gc.collect()
                return {"status": "Model unloaded. VRAM freed."}
            return {"status": "No model loaded."}

        elif action == "load":
            if not target_model:
                raise HTTPException(status_code=400, detail="Provide target_model filename.")
            model_path = os.path.join(MODELS_DIR, target_model)
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail="Model file not found.")
            
            # Unload existing model first
            if state.model:
                del state.model
                gc.collect()
            
            # Load new model, offloading to GPU (n_gpu_layers=-1 means all layers)
            state.model = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096)
            state.model_name = target_model
            return {"status": f"Loaded {target_model}"}

        elif action == "delete":
            if not target_model:
                raise HTTPException(status_code=400, detail="Provide target_model filename.")
            model_path = os.path.join(MODELS_DIR, target_model)
            if state.model_name == target_model:
                raise HTTPException(status_code=400, detail="Unload model before deleting.")
            if os.path.exists(model_path):
                os.remove(model_path)
                return {"status": f"Deleted {target_model}"}
            raise HTTPException(status_code=404, detail="File not found.")

        raise HTTPException(status_code=400, detail="Invalid action. Use load, unload, or delete.")

    @app.post("/sdai/api/chat")
    def chat_completion(messages: List[Dict[str, str]] = Body(...), max_tokens: int = Body(512)):
        """Example body: [{"role": "user", "content": "Write a python script."}]"""
        if not state.model:
            raise HTTPException(status_code=400, detail="No model loaded. Use /sdai/api/action to load one.")
        
        try:
            response = state.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

script_callbacks.on_app_started.append(mount_sdai_api)