import sys
import concurrent.futures
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
import uvicorn

sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

app = FastAPI()

# Initialize model
disable_torch_init()
model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-16F'
model, processor, tokenizer = model_init(model_path)

# Set parameters
temperature = 0.1
top_p = 0.6
max_output_tokens = 50
timeout_seconds = 50  # Set your desired timeout here

class InferenceRequest(BaseModel):
    modal: str
    modal_path: str
    instruct: str

def run_inference(modal, modal_path, instruct):
    return mm_infer(
        processor[modal](modal_path), 
        instruct, 
        model=model, 
        tokenizer=tokenizer, 
        do_sample=True, 
        modal=modal, 
        temperature=temperature, 
        top_p=top_p, 
        max_length=max_output_tokens
    )

# @app.post("/inference/")
# async def inference(request: InferenceRequest):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(run_inference, request.modal, request.modal_path, request.instruct)
#         try:
#             output = future.result(timeout=timeout_seconds)
#             return {"output": output}
#         except concurrent.futures.TimeoutError:
#             raise HTTPException(status_code=408, detail=f"Inference timed out after {timeout_seconds} seconds")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        modal = data['modal']
        modal_path = data['modal_path']
        instruct = data['instruct']

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_inference, modal, modal_path, instruct)
            try:
                output = future.result(timeout=timeout_seconds)
                response = f"Model Response: {output}"
            except concurrent.futures.TimeoutError:
                response = f"Error: Inference timed out after {timeout_seconds} seconds"
        await websocket.send_text(response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
