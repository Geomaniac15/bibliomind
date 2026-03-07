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
    keywords = [w for w in words if len(w) > 2]

    return ' '.join(keywords[:5])

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

    cover_url = None

    if best_match:
        cover_id = best_match.get('cover_i')
        if cover_id:
            cover_url = f'https://covers.openlibrary.org/b/id/{cover_id}-L.jpg'

    if best_match and best_score > 50:  # threshold
        return {
            'title': best_match.get('title'),
            'author': best_match.get('author_name', ['Unknown'])[0],
            'year': best_match.get('first_publish_year'),
            'cover': cover_url
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

    h, w = image.shape[:2]

    # crop middle region
    image = image[int(h*0.45):int(h*0.9), int(w*0.1):int(w*0.9)]

    # use red channel
    b, g, r = cv2.split(image)

    # otsu threshold
    blur = cv2.GaussianBlur(r, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite('debug.png', thresh)

    text = pytesseract.image_to_string(thresh, 
                                       lang='eng', 
                                       config='--psm 6')

    
    print('OCR Raw:', text)

    cleaned_text = clean_ocr_text(text)

    query = extract_keywords(cleaned_text)

    print('Search Query:', query)

    book = search_openlibrary(query)

    return {
        'ocr_text': text,
        'cleaned_text': cleaned_text,
        'query': query,
        'matched_book': book
    }