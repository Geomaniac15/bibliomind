from fastapi import FastAPI, UploadFile
import pytesseract
import cv2
import numpy as np
import os 
import requests 

os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_ocr_text(text):
    lines = text.split('\n')

    # remove empty lines
    lines = [line.strip() for line in lines if line.strip() != '']

    # remove numbers like "1"
    lines = [line for line in lines if not line.isdigit()]

    # join into one search string
    cleaned = ' '.join(lines)

    return cleaned

def search_openlibrary(query):
    url = 'https://openlibrary.org/search.json'
    params = {'q': query}

    response = requests.get(url, params=params)
    data = response.json()

    if data['numFound'] > 0:
        book = data['docs'][0]

        return {
            'title': book.get('title'),
            'author': book.get('author_name', ['Unknown'])[0],
            'year': book.get('first_publish_year'),
        }
    
    return None

app = FastAPI()

@app.post("/scan")
async def scan_books(file: UploadFile):
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    text = pytesseract.image_to_string(image, lang='eng')

    cleaned_text = clean_ocr_text(text)

    book = search_openlibrary(cleaned_text)

    return {
        'ocr_text': text,
        'cleaned_text': cleaned_text,
        'matched_book': book
    }