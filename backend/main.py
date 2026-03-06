from fastapi import FastAPI, UploadFile
import pytesseract
import cv2
import numpy as np
import os 
import requests 
import re

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_ocr_text(text):
    lines = text.split('\n')

    lines = [line.strip() for line in lines if line.strip() != '']

    cleaned = ' '.join(lines)

    # remove numbers and weird characters
    cleaned = re.sub(r'[^A-Za-z\s]', '', cleaned)

    return cleaned

def search_openlibrary(query):
    url = 'https://openlibrary.org/search.json'
    params = {'title': query}

    try:
        response = requests.get(url, params=params, timeout=5)

        if response.status_code != 200:
            print('OpenLibrary error:', response.status_code)
            return None

        data = response.json()

    except requests.exceptions.RequestException as e:
        print('Network error:', e)
        return None

    except ValueError:
        print('Invalid JSON returned from API')
        return None

    if data.get('docs'):
        book = data['docs'][0]

        return {
            'title': book.get('title'),
            'author': book.get('author_name', ['Unknown'])[0],
            'year': book.get('first_publish_year')
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

    text = pytesseract.image_to_string(image, lang='eng')

    cleaned_text = clean_ocr_text(text)

    query_words = cleaned_text.split()

    queries = [
        ' '.join(query_words[:5]),
        ' '.join(query_words[:4]),
        ' '.join(query_words[:3]),
        ' '.join(query_words[:2])
    ]

    book = None

    for q in queries:
        book = search_openlibrary(q)
        if book:
            query = q
            break

    if not book:
        query = queries[0]

    return {
        'ocr_text': text,
        'cleaned_text': cleaned_text,
        'query': query,
        'matched_book': book
    }