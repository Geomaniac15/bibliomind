from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import requests 
import re
from rapidfuzz import fuzz
import easyocr
from spellchecker import SpellChecker

spell = SpellChecker()

print('loading EasyOCR model...')
reader = easyocr.Reader(['en'])
print('model loaded')

def correct_ocr_text(text):
    words = text.split()
    corrected = []
    for word in words:
        correction = spell.correction(word)
        corrected.append(correction if correction else word)
    return ' '.join(corrected)

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

def dynamic_book_crop(image):
    # 1. convert to greyscale and blur to remove background static
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # 2. canny edge detection
    edged = cv2.Canny(blur, 50, 150)

    # 3. find shapes these edges make
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image 

    # 4. sort the shapes by size, largest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # look at the biggest shape
    largest_contour = contours[0]
    
    # get the strict bounding box (x, y, width, height) of that shape
    x, y, w, h = cv2.boundingRect(largest_contour)

    img_area = image.shape[0] * image.shape[1]
    if (w * h) > (0.15 * img_area):
        return image[y:y+h, x:x+w]

    img_h, img_w = image.shape[:2]
    return image[int(img_h*0.15):int(img_h*0.95), int(img_w*0.2):int(img_w*0.8)]

app = FastAPI()

@app.post('/scan')
async def scan_books(file: UploadFile):
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    h, w = image.shape[:2]

    cropped_image = image[int(h*0.15):int(h*0.95), int(w*0.2):int(w*0.8)]

    upscaled = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(upscaled)
    
    results.sort(key=lambda item: item[0][0][1])

    large_text_blocks = []
    
    img_height = upscaled.shape[0]
    
    threshold = img_height * 0.03  # 3% of image height instead of fixed 50px

    for (bbox, text, prob) in results:
        text_height = bbox[3][1] - bbox[0][1]
        if text_height > threshold and prob > 0.3:  # also add confidence filter
            large_text_blocks.append(text)
    
    if not large_text_blocks:
        large_text_blocks = [text for (_, text, prob) in results if prob > 0.3]
            
    text = ' '.join(large_text_blocks)
    print('Filtered OCR Text:', text)

    cleaned_text = clean_ocr_text(text)
    corrected_text = correct_ocr_text(cleaned_text)
    query = extract_keywords(corrected_text)
    print('Search Query:', query)

    book = search_openlibrary(query)

    return {
        'ocr_text': text,
        'cleaned_text': cleaned_text,
        'query': query,
        'matched_book': book
    }