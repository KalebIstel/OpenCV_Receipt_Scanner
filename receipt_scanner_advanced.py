#!/usr/bin/env python3
"""
receipt_scanner_advanced.py – CV enhancements for the receipt scanner.
Adds: Hough‑line deskew, ROI‑based item extraction, currency detection, Flask API.
"""

import cv2
import numpy as np
import re
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import the original scanner (assumes receipt_scanner.py is in same folder)
from receipt_scanner import ReceiptPreprocessor, ReceiptOCR, ReceiptDatabase, ReceiptScanner

# =============================================================================
# 1. ADVANCED DESKEW USING HOUGH LINES (more accurate than contour‑based)
# =============================================================================
def deskew_with_hough(img: np.ndarray) -> np.ndarray:
    """
    Compute rotation angle from dominant Hough lines and deskew the image.
    Works better on receipts with horizontal text lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is None:
        return img
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return img
    
    median_angle = np.median(angles)
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return rotated

# =============================================================================
# 2. ITEM LINE SEGMENTATION USING HORIZONTAL PROJECTION AND TABLE DETECTION
# =============================================================================
class LineItemExtractor:
    """
    Extracts individual line items from a preprocessed receipt image.
    Uses horizontal projection to find text rows, then isolates price columns.
    """
    def __init__(self, debug=False):
        self.debug = debug
        
    def _horizontal_projection(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous row segments where text is present."""
        h = binary.shape[0]
        proj = np.sum(binary == 0, axis=1)  # count black pixels (text)
        threshold = np.mean(proj) * 0.2
        rows = proj > threshold
        
        segments = []
        start = None
        for i in range(h):
            if rows[i] and start is None:
                start = i
            elif not rows[i] and start is not None:
                if i - start > 10:   # ignore isolated noise
                    segments.append((start, i))
                start = None
        if start is not None and h - start > 10:
            segments.append((start, h))
        return segments
    
    def _split_into_columns(self, img_bgr: np.ndarray, y_start: int, y_end: int, 
                            expected_columns=2) -> List[str]:
        """
        For a given row band, use vertical projection to split description and price.
        Returns list of text strings from that row.
        """
        roi = img_bgr[y_start:y_end, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Vertical projection
        v_proj = np.sum(binary, axis=0) > 100
        # Find gaps of white space (no text) that separate columns
        gaps = []
        in_gap = False
        for x in range(len(v_proj)):
            if not v_proj[x] and not in_gap:
                in_gap = True
                start_gap = x
            elif v_proj[x] and in_gap:
                in_gap = False
                if x - start_gap > 30:   # gap width threshold
                    gaps.append((start_gap, x))
        
        # Use the largest two gaps as column separators
        if len(gaps) >= 2:
            gaps.sort(key=lambda g: g[1]-g[0], reverse=True)
            split_x = gaps[0][0] + (gaps[0][1]-gaps[0][0])//2
        else:
            # fallback: assume price is in the rightmost 1/3 of the image
            split_x = int(roi.shape[1] * 0.7)
        
        # Extract description and price regions
        desc_roi = roi[:, :split_x]
        price_roi = roi[:, split_x:]
        
        # Use Tesseract on each sub‑region (lightweight, reuses existing OCR)
        import pytesseract
        desc_text = pytesseract.image_to_string(desc_roi, config='--psm 7').strip()
        price_text = pytesseract.image_to_string(price_roi, config='--psm 7 --psm 8').strip()
        
        # Extract the first price‑like pattern
        price_match = re.search(r'(\d+[.,]\d{2})', price_text)
        return [desc_text, price_match.group(1) if price_match else ""]
    
    def extract_items(self, img_bgr: np.ndarray) -> List[Dict]:
        """
        Main method: returns list of {'description': str, 'price': float}
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row_segments = self._horizontal_projection(binary)
        
        items = []
        for (y1, y2) in row_segments:
            # skip very small rows (likely noise)
            if y2 - y1 < 12:
                continue
            # skip header/footer regions by checking typical price pattern
            roi_text = self._split_into_columns(img_bgr, y1, y2)
            if len(roi_text) >= 2 and roi_text[1]:
                try:
                    price = float(roi_text[1].replace(',', '.'))
                    # filter obvious non‑items (e.g., totals, tax)
                    lower_desc = roi_text[0].lower()
                    if not any(keyword in lower_desc for keyword in ['total', 'tax', 'subtotal', 'vat']):
                        items.append({
                            'description': roi_text[0],
                            'price': price
                        })
                except ValueError:
                    continue
        return items

# =============================================================================
# 3. CURRENCY DETECTION (AUD, USD, EUR, GBP, JPY)
# =============================================================================
def detect_currency(text: str) -> str:
    """Return ISO currency code based on symbols or words in the receipt."""
    text = text.upper()
    if 'AUD' in text or 'AU $' in text or 'A$' in text:
        return 'AUD'
    if 'EUR' in text or '€' in text:
        return 'EUR'
    if 'GBP' in text or '£' in text:
        return 'GBP'
    if 'JPY' in text or '¥' in text:
        return 'JPY'
    if 'USD' in text or 'US $' in text or '$' in text:
        return 'USD'
    return 'UNKNOWN'

# =============================================================================
# 4. FLASK API FOR REMOTE SCANNING
# =============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path.cwd() / "tmp" / "receipt_uploads"
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

@app.route('/scan', methods=['POST'])
def api_scan_receipt():
    """Upload a receipt image and receive structured data."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save temporarily
    filename = secure_filename(file.filename)
    save_path = app.config['UPLOAD_FOLDER'] / filename
    file.save(save_path)
    
    # Run the scanner (we extend the original scanner to use our improved deskew & item extraction)
    scanner = ReceiptScanner(debug=False)
    # We'll override the deskew step with Hough‑based version
    
    def enhanced_process(source, auto_detect=True):
        img = scanner.preprocessor.load_image(source)
        if auto_detect:
            pts = scanner.preprocessor.detect_document_contour(img)
            if pts is not None:
                img = scanner.preprocessor.perspective_transform(img, pts)
        img = deskew_with_hough(img)   # <-- better deskew
        processed = scanner.preprocessor.enhance_for_ocr(img)
        return processed
    
    scanner.preprocessor.process = enhanced_process
    data = scanner.scan(save_path)
    
    # Enhance item list with line‑by‑line extraction
    item_extractor = LineItemExtractor()
    img_for_items = cv2.imread(str(save_path))
    if img_for_items is not None:
        better_items = item_extractor.extract_items(img_for_items)
        if better_items:
            data['items'] = better_items   # replace coarse regex items
            data['items_json'] = json.dumps(better_items)
            # Avoid duplicate inserts here; scanner.scan() already wrote one record.
    
    data['currency'] = detect_currency(data.get('raw_text', ''))
    
    # Clean up temp file
    save_path.unlink(missing_ok=True)
    
    return jsonify({
        'receipt_id': data.get('receipt_id'),
        'vendor': data.get('vendor'),
        'date': data.get('date'),
        'total': data.get('total'),
        'currency': data['currency'],
        'items': data.get('items', []),
        'confidence': data.get('confidence')
    })

@app.route('/stats', methods=['GET'])
def api_stats():
    """Return spending statistics from the database."""
    scanner = ReceiptScanner()
    stats = scanner.stats()
    return jsonify(stats)

@app.route('/search', methods=['GET'])
def api_search():
    """Search receipts by vendor, date range, amount."""
    scanner = ReceiptScanner()
    vendor = request.args.get('vendor')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    min_amount = request.args.get('min_amount')
    max_amount = request.args.get('max_amount')
    results = scanner.query(vendor=vendor, date_from=date_from, date_to=date_to,
                            min_amount=float(min_amount) if min_amount else None,
                            max_amount=float(max_amount) if max_amount else None)
    return jsonify(results)

def start_api(host='0.0.0.0', port=5000):
    """Start the Flask API server (run from command line)."""
    app.run(host=host, port=port, debug=False)

# =============================================================================
# 5. COMMAND LINE INTERFACE FOR NEW FEATURES
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Receipt Scanner")
    parser.add_argument("--api", action="store_true", help="Start REST API server")
    parser.add_argument("--hough-deskew", help="Apply Hough deskew to an image and save", metavar="IMAGE")
    parser.add_argument("--extract-items", help="Extract line items from an image", metavar="IMAGE")
    parser.add_argument("--currency", help="Detect currency from an image", metavar="IMAGE")
    args = parser.parse_args()
    
    if args.api:
        print("Starting Receipt Scanner API on http://0.0.0.0:5000")
        start_api()
    elif args.hough_deskew:
        img = cv2.imread(args.hough_deskew)
        if img is None:
            raise ValueError(f"Could not load image: {args.hough_deskew}")
        out = deskew_with_hough(img)
        out_path = f"hough_deskewed_{Path(args.hough_deskew).name}"
        cv2.imwrite(out_path, out)
        print(f"Saved deskewed image to {out_path}")
    elif args.extract_items:
        img = cv2.imread(args.extract_items)
        if img is None:
            raise ValueError(f"Could not load image: {args.extract_items}")
        extractor = LineItemExtractor(debug=True)
        items = extractor.extract_items(img)
        print(json.dumps(items, indent=2))
    elif args.currency:
        import pytesseract
        img = cv2.imread(args.currency)
        if img is None:
            raise ValueError(f"Could not load image: {args.currency}")
        text = pytesseract.image_to_string(img)
        print(f"Detected currency: {detect_currency(text)}")
    else:
        parser.print_help()