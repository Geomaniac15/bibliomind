from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import requests
import re
import sqlite3
from datetime import datetime
from rapidfuzz import fuzz
import easyocr

print('loading EasyOCR model...')
reader = easyocr.Reader(['en'])
print('model loaded')

# database

def get_db():
    conn = sqlite3.connect('bibliomind.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            author          TEXT,
            year            INTEGER,
            cover_url       TEXT,
            subjects        TEXT,          -- comma-separated tags from OpenLibrary
            owned           INTEGER DEFAULT 1,  -- 1=owned, 0=wanted
            status          TEXT DEFAULT 'unread', -- unread/reading/read/dnf
            rating          INTEGER,       -- 1-5, nullable
            notes           TEXT,
            date_added      TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# OCR helpers

OCR_CHAR_FIXES = [
    (r'\b1\b', 'I'),
    (r'\bIl([a-z])', r'H\1'),
    (r'([A-Za-z])tc([a-z])', r'\1ve\2'),
    (r'\bNc([a-z])', r'Ne\1'),
    (r'([a-z])c\b', r'\1e'),
    (r'\bMc([a-z])', r'Me\1'),
    (r'\bMarp', 'Harp'),
    (r'\bWIo\b', 'Who'),
]

def fix_ocr_errors(text):
    for pattern, replacement in OCR_CHAR_FIXES:
        text = re.sub(pattern, replacement, text)
    return text

def preprocess_for_ocr(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def clean_ocr_text(text):
    cleaned = re.sub(r'[^A-Za-z\s]', '', text)
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    return ' '.join(lines)

def split_title_author(text):
    words = text.split()
    if len(words) >= 6:
        return ' '.join(words[:-2]), ' '.join(words[-2:])
    return text, None

# openlibrary

def search_openlibrary(title_query):
    url = 'https://openlibrary.org/search.json'

    response = requests.get(url, params={'title': title_query.lower()})
    data = response.json()

    best_match, best_score = None, 0

    for doc in data.get('docs', [])[:10]:
        title = doc.get('title', '')
        score = fuzz.token_sort_ratio(title_query.lower(), title.lower())
        if score > best_score:
            best_score = score
            best_match = doc

    # fallback: broad q= search
    if not best_match or best_score <= 50:
        response = requests.get(url, params={'q': title_query.lower()})
        data = response.json()
        for doc in data.get('docs', [])[:10]:
            title = doc.get('title', '')
            score = fuzz.token_sort_ratio(title_query.lower(), title.lower())
            if score > best_score:
                best_score = score
                best_match = doc

    if not best_match or best_score <= 50:
        return None

    cover_url = None
    cover_id = best_match.get('cover_i')
    if cover_id:
        cover_url = f'https://covers.openlibrary.org/b/id/{cover_id}-L.jpg'

    subjects = best_match.get('subject', [])[:10]

    return {
        'title':    best_match.get('title'),
        'author':   best_match.get('author_name', ['Unknown'])[0],
        'year':     best_match.get('first_publish_year'),
        'cover':    cover_url,
        'subjects': subjects,
    }

# request models

class SaveBookRequest(BaseModel):
    title:    str
    author:   Optional[str] = None
    year:     Optional[int] = None
    cover:    Optional[str] = None
    subjects: Optional[list[str]] = []
    owned:    bool = True
    status:   str = 'unread'   # unread / reading / read / dnf
    rating:   Optional[int] = None  # 1-5
    notes:    Optional[str] = None

class UpdateBookRequest(BaseModel):
    owned:  Optional[bool] = None
    status: Optional[str] = None
    rating: Optional[int] = None
    notes:  Optional[str] = None

# app

app = FastAPI()

@app.post('/scan')
async def scan_book(file: UploadFile):
    '''OCR scan + OpenLibrary lookup. Returns preview — does NOT save.'''
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    h, w = image.shape[:2]
    cropped = image[int(h * 0.15):int(h * 0.95), int(w * 0.2):int(w * 0.8)]
    upscaled = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    upscaled = preprocess_for_ocr(upscaled)

    results = reader.readtext(upscaled)
    results.sort(key=lambda item: item[0][0][1])

    img_height = upscaled.shape[0]
    threshold = img_height * 0.03

    large_text_blocks = [
        text for (bbox, text, prob) in results
        if (bbox[3][1] - bbox[0][1]) > threshold and prob > 0.3
    ]
    if not large_text_blocks:
        large_text_blocks = [text for (_, text, prob) in results if prob > 0.3]

    raw_text = ' '.join(large_text_blocks)
    fixed_text = fix_ocr_errors(raw_text)
    cleaned_text = clean_ocr_text(fixed_text)
    title_q, _ = split_title_author(cleaned_text)

    book = search_openlibrary(title_q)

    return {
        'ocr_text':     raw_text,
        'matched_book': book  # None if no match — frontend should handle this
    }


@app.post('/books')
def save_book(body: SaveBookRequest):
    '''Save a confirmed book to the database.'''
    conn = get_db()

    existing = conn.execute(
        'SELECT id FROM books WHERE LOWER(title) = LOWER(?) AND LOWER(COALESCE(author,'')) = LOWER(?)',
        (body.title, body.author or '')
    ).fetchone()

    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail='Book already in your library')

    conn.execute('''
        INSERT INTO books (title, author, year, cover_url, subjects, owned, status, rating, notes, date_added)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        body.title,
        body.author,
        body.year,
        body.cover,
        ','.join(body.subjects) if body.subjects else '',
        1 if body.owned else 0,
        body.status,
        body.rating,
        body.notes,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    book_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    conn.close()

    return {'id': book_id, 'message': 'Book saved'}


@app.get('/books')
def list_books(status: Optional[str] = None, owned: Optional[bool] = None):
    '''list all books, optionally filtered by status or owned/wanted.'''
    conn = get_db()
    query = 'SELECT * FROM books WHERE 1=1'
    params = []

    if status:
        query += ' AND status = ?'
        params.append(status)
    if owned is not None:
        query += ' AND owned = ?'
        params.append(1 if owned else 0)

    query += ' ORDER BY date_added DESC'
    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(row) for row in rows]


@app.patch('/books/{book_id}')
def update_book(book_id: int, body: UpdateBookRequest):
    '''update rating, notes, status, or owned flag.'''
    conn = get_db()

    book = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    if not book:
        conn.close()
        raise HTTPException(status_code=404, detail='Book not found')

    fields, params = [], []
    if body.owned is not None:
        fields.append('owned = ?')
        params.append(1 if body.owned else 0)
    if body.status is not None:
        if body.status not in ('unread', 'reading', 'read', 'dnf'):
            raise HTTPException(status_code=400, detail='Invalid status')
        fields.append('status = ?')
        params.append(body.status)
    if body.rating is not None:
        if not 1 <= body.rating <= 5:
            raise HTTPException(status_code=400, detail='Rating must be 1-5')
        fields.append('rating = ?')
        params.append(body.rating)
    if body.notes is not None:
        fields.append('notes = ?')
        params.append(body.notes)

    if fields:
        params.append(book_id)
        conn.execute(f'UPDATE books SET {', '.join(fields)} WHERE id = ?', params)
        conn.commit()

    conn.close()
    return {'message': 'Updated'}


@app.delete('/books/{book_id}')
def delete_book(book_id: int):
    conn = get_db()
    book = conn.execute('SELECT id FROM books WHERE id = ?', (book_id,)).fetchone()
    if not book:
        conn.close()
        raise HTTPException(status_code=404, detail='Book not found')
    conn.execute('DELETE FROM books WHERE id = ?', (book_id,))
    conn.commit()
    conn.close()
    return {'message': 'Deleted'}


@app.get('/recommendations/{book_id}')
def recommend(book_id: int, limit: int = 5):
    '''recommend books based on the subjects of a saved book.'''
    conn = get_db()
    book = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    if not book:
        conn.close()
        raise HTTPException(status_code=404, detail='Book not found')

    subjects = [s.strip() for s in (book['subjects'] or '').split(',') if s.strip()]
    if not subjects:
        conn.close()
        raise HTTPException(status_code=400, detail='No subjects stored for this book — cannot recommend')

    owned_titles = {
        row['title'].lower()
        for row in conn.execute('SELECT title FROM books').fetchall()
    }
    conn.close()

    query = ' '.join(subjects[:2])
    response = requests.get('https://openlibrary.org/search.json', params={'subject': query, 'limit': 20})
    data = response.json()

    recommendations = []
    for doc in data.get('docs', []):
        title = doc.get('title', '')
        if title.lower() in owned_titles:
            continue

        cover_id = doc.get('cover_i')
        recommendations.append({
            'title':  title,
            'author': doc.get('author_name', ['Unknown'])[0],
            'year':   doc.get('first_publish_year'),
            'cover':  f'https://covers.openlibrary.org/b/id/{cover_id}-L.jpg' if cover_id else None,
        })

        if len(recommendations) >= limit:
            break

    return {
        'based_on':       {'title': book['title'], 'subjects': subjects[:2]},
        'recommendations': recommendations
    }