from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import requests 
import re
from rapidfuzz import fuzz
import easyocr

print('loading EasyOCR model...')
reader = easyocr.Reader(['en'])
print('model loaded')

def clean_ocr_text(text):
    # remove numbers and weird characters
    cleaned = re.sub(r'[^A-Za-z\s]', '', text)
    lines = cleaned.split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']
    return ' '.join(lines)

def extract_keywords(text):
    words = text.upper().split()
    
    # filter out common publisher hype words
    ignore_words = {'SUNDAY', 'TIMES', 'BESTSELLER', 'NEW', 'YORK', 'AUTHOR', 'NUMBER'}
    
    # keep words longer than 2 characters that aren't in the ignore list
    keywords = [w for w in words if len(w) > 2 and w not in ignore_words]
    
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

    # crop
    cropped_image = image[int(h*0.50):int(h*0.9), int(w*0.1):int(w*0.9)]

    results = reader.readtext(cropped_image, detail=0)
    
    # join the list into a single string
    text = ' '.join(results)
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