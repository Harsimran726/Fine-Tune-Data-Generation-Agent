import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from Data_Geneartion_Agent import generate_data_agent
import os
from dotenv import load_dotenv

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your Netlify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    api_key: str
    query: str

class Result(BaseModel):
    status: str
    message: str
    csv_file: Optional[str] = None

def save_api_key(api_key: str):
    """Save API key to .env file"""
    with open(".env", "w") as f:
        f.write(f"GOOGLE_API_KEY={api_key}\n")
    # Reload environment variables
    load_dotenv()

@app.get("/",response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

# class ApiKey(BaseModel):
#     api_key: str

# @app.post("/api/save_api_key")
# async def save_api_key(api_key: ApiKey):
#     with open(".env", "w") as f:
#         f.write(f"GOOGLE_API_KEY={api_key.api_key}")
#     return JSONResponse(content={"message": "API key saved successfully"}, status_code=200)

@app.post("/generate")
async def generate(query: Query):
    try:
        # Save the API key to .env file
        save_api_key(query.api_key)
        
        # Generate data using the agent with the new API key
        result = generate_data_agent(query.query)
        print(f"Here is the final result {result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        print(f"Here is file filename in main.py  {filename}")
        return FileResponse(
            path=filename,
            filename=filename,
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
