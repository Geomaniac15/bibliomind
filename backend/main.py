from fastapi import FastAPI, UploadFile
import pytesseract
import cv2
import numpy as np
import os 
import requests 
import re
from rapidfuzz import fuzz

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_ocr_text(text):
    lines = text.split('\n')

    lines = [line.strip() for line in lines if line.strip() != '']

    cleaned = ' '.join(lines)

    # remove numbers and weird characters
    cleaned = re.sub(r'[^A-Za-z\s]', '', cleaned)

    return cleaned

def extract_keywords(text):
    words = text.split()

    # keep longer uppercase tokens
    keywords = [w for w in words if len(w) > 4]

    return ' '.join(keywords[:3])

def search_openlibrary(query):
    url = 'https://openlibrary.org/search.json'
    params = {'q': query}

    response = requests.get(url, params=params)
    data = response.json()

    best_match = None
    best_score = 0

    for doc in data.get('docs', [])[:10]:  # check first few results
        title = doc.get('title', '')

        score = fuzz.partial_ratio(query.lower(), title.lower())

        if score > best_score:
            best_score = score
            best_match = doc

    if best_match and best_score > 60:  # threshold
        return {
            'title': best_match.get('title'),
            'author': best_match.get('author_name', ['Unknown'])[0],
            'year': best_match.get('first_publish_year')
        }

    return None

app = FastAPI()

@app.post('/scan')
async def scan_books(file: UploadFile):
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # increase constrast
    grey = cv2.GaussianBlur(grey, (3,3), 0)

    # threshold makes text stand out
    thresh = cv2.adaptiveThreshold(
        grey, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')

    cleaned_text = clean_ocr_text(text)

    query = extract_keywords(cleaned_text)

    book = search_openlibrary(query)

    return {
        'ocr_text': text,
        'cleaned_text': cleaned_text,
        'query': query,
        'matched_book': book
    }