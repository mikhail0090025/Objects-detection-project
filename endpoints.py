from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
import numpy as np

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import neural_net as nn
import io
from PIL import Image, ImageDraw
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
        return JSONResponse(content=response, status_code=200, media_type="application/json")
    except ValueError as e:
        return Response(f"Request error: {e}", 400)
    except Exception as e:
        return Response(f"Internal server error: {e}", 500)

@app.get("/get_graphics")
def get_graphics_endpoint():
    try:
        # Создаём график
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in range(len(nn.all_losses))],
            y=nn.all_losses,
            mode='lines+markers',
            name='Loss',
            fill='tozeroy',
        ))
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in range(len(nn.all_val_losses))],
            y=nn.all_val_losses,
            mode='lines+markers',
            name='Validation loss',
            fill='tozeroy',
        ))
        fig.update_layout(
            title="Loss Graph",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
            yaxis=dict(range=[0, None])
        )

        # Конвертируем в JSON
        graph_json = fig.to_json()
        return {"graph": graph_json}
    except Exception as e:
        return Response(f"Unexpected error has occured: {e}")

from torchvision import transforms
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import requests
from io import BytesIO
import torch

# @app.get("/process_image")
# async def process_image(url: str):
#     try:
#         transform = transforms.Compose([
#             transforms.Resize((200, 200)),  # Сжатие до 200x200
#             transforms.ToTensor(),          # Преобразование в тензор
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация в [-1, 1]
#         ])
#         # Скачиваем изображение по URL
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()  # Проверяем, что запрос успешен
#         img = Image.open(io.BytesIO(response.content)).convert('RGB')
# 
#         # Преобразуем в тензор
#         img_tensor = transform(img).unsqueeze(0).to(nn.device)  # (1, 3, 200, 200)
# 
#         # Прогоняем через модель
#         with nn.torch.no_grad():
#             output_tensor = nn.detector(img_tensor)  # (1, 3, 200, 200)
# 
#         return JSONResponse(content=output_tensor.squeeze(0).cpu().tolist())
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/process_image")
async def process_image(url: str):
    try:
        from my_dataset_handler import start_size
        # Определяем трансформации
        transform = transforms.Compose([
            transforms.Resize(start_size),  # Сжатие до 200x200
            transforms.ToTensor(),          # Преобразование в тензор
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация в [-1, 1]
        ])

        # Скачиваем изображение по URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Проверяем, что запрос успешен
        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Сохраняем оригинальное изображение для дорисовки
        img_with_boxes = img.copy()

        # Преобразуем в тензор
        img_tensor = transform(img).unsqueeze(0).to(nn.device)  # (1, 3, 200, 200)

        # Прогоняем через модель
        with torch.no_grad():
            output_tensor = nn.detector(img_tensor)  # Должно быть (1, max_objects_per_img, 4)

        # Извлекаем координаты (norm_xmin, norm_ymin, norm_xmax, norm_ymax)
        boxes = output_tensor.squeeze(0).cpu().numpy()  # (max_objects_per_img, 4)
        draw = ImageDraw.Draw(img_with_boxes)

        # Дорисовываем квадраты
        for box in boxes:
            if box.sum() > 0:  # Проверяем, не нулевой ли бокс
                xmin, ymin, xmax, ymax = box
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                # Преобразуем нормализованные координаты в пиксели
                xmin_px = int(xmin * img.width)
                ymin_px = int(ymin * img.height)
                xmax_px = int(xmax * img.width)
                ymax_px = int(ymax * img.height)
                # Рисуем прямоугольник
                draw.rectangle([xmin_px, ymin_px, xmax_px, ymax_px], outline="red", width=2)

        # Преобразуем изображение обратно в байты для возврата
        img_byte_arr = BytesIO()
        img_with_boxes.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Возвращаем изображение и координаты
        response_data = {
            "text_answer": "Image processed with bounding boxes",
            "boxes": boxes.tolist()  # Возвращаем координаты для клиента
        }
        return StreamingResponse(
            content=img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=processed_image.png"}
        )# , JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    import os
    os.system("uvicorn endpoints:app --host 0.0.0.0 --port 5000")