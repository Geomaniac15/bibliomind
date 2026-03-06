from fastapi import FastAPI, UploadFile
import pytesseract
import cv2
import numpy as np
import os 

os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdate"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

@app.post("/scan")
async def scan_books(file: UploadFile):
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )
    text = pytesseract.image_to_string(image)

    return {'detected_text': text}