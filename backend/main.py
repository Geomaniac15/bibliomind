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

# common serif font OCR misreads
OCR_CHAR_FIXES = [
    (r'\b1\b', 'I'),        # standalone 1 → I
    (r'\bIl([a-z])', r'H\1'), # Il... → H... (e.g. Ilavc → Have)
    (r'([A-Za-z])tc([a-z])', r'\1ve\2'),  # tc → ve (Nctcr → Never)
    (r'\bNc([a-z])', r'Ne\1'),  # Nc... → Ne...
    (r'([a-z])c\b', r'\1e'), # trailing c → e (Mcn → Men)
    (r'\bMc([a-z])', r'Me\1'), # Mc... → Me... (Mcn → Men)
    (r'arp', 'arp'),         # keep arp (Marpman → Harpman needs H fix)
    (r'\bMarp', 'Harp'),     # Marpman → Harpman
    (r'\bWIo\b', 'Who'),     # specific misread
]

def fix_ocr_errors(text):
    for pattern, replacement in OCR_CHAR_FIXES:
        text = re.sub(pattern, replacement, text)
    return text

def preprocess_for_ocr(image):
    # boost contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    
    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)
    
    return image

def clean_ocr_text(text):
    # remove numbers and weird characters
    cleaned = re.sub(r'[^A-Za-z\s]', '', text)
    lines = cleaned.split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']
    return ' '.join(lines)

def extract_keywords(text):
    words = text.upper().split()
    ignore_words = {'SUNDAY', 'TIMES', 'BESTSELLER', 'NEW', 'YORK', 'AUTHOR', 'NUMBER', 'THE'}
    
    def looks_clean(w):
        return w.isalpha() and len(w) >= 3 and w not in ignore_words

    clean_words = [w for w in words if looks_clean(w)]
    return ' '.join(clean_words[:5])

def split_title_author(fixed_text):
    words = fixed_text.split()
    # heuristic: if 6+ words, last 2 are probably author firstname/lastname
    if len(words) >= 6:
        return ' '.join(words[:-2]), ' '.join(words[-2:])
    return fixed_text, None

def search_openlibrary(title_query, author_query=None):
    url = 'https://openlibrary.org/search.json'
    params = {'title': title_query}
    if author_query:
        params['author'] = author_query

    response = requests.get(url, params=params)
    data = response.json()

    best_match = None
    best_score = 0

    for doc in data.get('docs', [])[:10]:
        title = doc.get('title', '')
        score = fuzz.token_sort_ratio(title_query.lower(), title.lower())

        if score > best_score:
            best_score = score
            best_match = doc

    # if no good match, retry with just the longest clean word
    if (not best_match or best_score <= 50) and title_query:
        longest = sorted(title_query.split(), key=len, reverse=True)[0]
        if longest != title_query:
            params = {'q': longest}
            response = requests.get(url, params=params)
            data = response.json()
            for doc in data.get('docs', [])[:10]:
                title = doc.get('title', '')
                score = fuzz.token_sort_ratio(longest.lower(), title.lower())
                if score > best_score:
                    best_score = score
                    best_match = doc

    cover_url = None
    if best_match:
        cover_id = best_match.get('cover_i')
        if cover_id:
            cover_url = f'https://covers.openlibrary.org/b/id/{cover_id}-L.jpg'

    if best_match and best_score > 50:
        return {
            'title': best_match.get('title'),
            'author': best_match.get('author_name', ['Unknown'])[0],
            'year': best_match.get('first_publish_year'),
            'cover': cover_url
        }

    return None

def dynamic_book_crop(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image 

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
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
    upscaled = preprocess_for_ocr(upscaled)

    results = reader.readtext(upscaled)
    results.sort(key=lambda item: item[0][0][1])

    large_text_blocks = []
    img_height = upscaled.shape[0]
    threshold = img_height * 0.03

    for (bbox, text, prob) in results:
        text_height = bbox[3][1] - bbox[0][1]
        if text_height > threshold and prob > 0.3:
            large_text_blocks.append(text)
    
    if not large_text_blocks:
        large_text_blocks = [text for (_, text, prob) in results if prob > 0.3]
            
    text = ' '.join(large_text_blocks)
    print('Filtered OCR Text:', text)

    cleaned_text = clean_ocr_text(text)
    fixed_text = fix_ocr_errors(cleaned_text)   # char-level fixes, no spell checker
    title_q, author_q = split_title_author(fixed_text)
    
    query = extract_keywords(title_q)
    print('Search Query:', query)

    book = search_openlibrary(query, author_q)

    return {
        'ocr_text': text,
        'cleaned_text': cleaned_text,
        'fixed_text': fixed_text,
        'query': query,
        'matched_book': book
    }