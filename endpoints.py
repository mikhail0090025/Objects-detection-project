from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
import jinja2

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import neural_net as nn
import io
from PIL import Image
import plotly.graph_objects as go


app = FastAPI()

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root(request: Request):
    try:
        return templates.TemplateResponse("mainpage.html", {"request": request})
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.get("/mainpage")
def root_redirect():
    return RedirectResponse("/")

@app.get("/one_epoch")
def one_epoch_endpoint():
    try:
        loss, val_loss = nn.one_epoch()
        response = {
            "text_answer": "One epoch passed",
            "loss": loss,
            "val_loss": val_loss,
        }
        nn.show_graphics()
        return JSONResponse(content=response, status_code=200, media_type="application/json")
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.get("/go_epochs/{epochs_count}")
def one_epoch_endpoint(epochs_count: str):
    try:
        epochs_count = int(epochs_count)
        loss, val_loss = nn.go_epochs(epochs_count)
        response = {
            "text_answer": f"{epochs_count} epoch{'s' if epochs_count > 1 else ''} passed",
            "loss": loss,
            "val_loss": val_loss,
        }
        nn.show_graphics()
        return JSONResponse(content=response, status_code=200, media_type="application/json")
    except ValueError as e:
        return Response(f"Request error: {e}", 400)
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

if __name__ == '__main__':
    import os
    os.system("uvicorn endpoints:app --host 0.0.0.0 --port 5000")