#!/usr/bin/env python3
"""
Treasurer Receipt Scanner / Logger
====================================
A complete receipt digitization pipeline using OpenCV for document preprocessing
and Tesseract OCR for text extraction, with SQLite storage.

Corrections made:
- Fixed cv2.minAreaRect() unpacking in deskew()
- Changed text.split('\\n') to text.split('\n')
- Fixed all regex patterns (removed double backslashes in raw strings)
- Added explicit handling for missing Tesseract
"""

import cv2
import numpy as np
import sqlite3
import json
import re
import os
import sys
import csv
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Optional: pytesseract for OCR (install: pip install pytesseract)
# Also requires Tesseract OCR engine installed on system
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("WARNING: pytesseract not installed. OCR functionality disabled.")
    print("Install with: pip install pytesseract")


class ReceiptPreprocessor:
    """
    OpenCV-based document preprocessing pipeline.
    Handles perspective correction, deskewing, and image enhancement.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.debug_dir = Path("debug_output")
        if debug:
            self.debug_dir.mkdir(exist_ok=True)

    def load_image(self, source) -> np.ndarray:
        """Load image from file path, camera, or numpy array."""
        if isinstance(source, str) or isinstance(source, Path):
            img = cv2.imread(str(source))
            if img is None:
                raise ValueError(f"Could not load image from {source}")
            return img
        elif isinstance(source, np.ndarray):
            return source.copy()
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

    def capture_from_camera(self, camera_id: int = 0,
                           warmup_frames: int = 5) -> np.ndarray:
        """Capture image from webcam."""
        backends = [None]
        if os.name == "nt":
            # On Windows, MSMF can fail on some webcams; DSHOW is a common fallback.
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        last_error = "Unknown camera error"
        for backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, backend)

            if not cap.isOpened():
                cap.release()
                last_error = f"Could not open camera {camera_id} (backend={backend})"
                continue

            # Warmup: discard initial frames for auto-exposure
            for _ in range(warmup_frames):
                cap.read()

            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                return frame

            last_error = f"Failed to capture frame from camera {camera_id} (backend={backend})"

        raise RuntimeError(last_error)

    def detect_document_contour(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest quadrilateral contour (receipt/document).
        Returns 4 corner points or None if no document found.
        """
        # Resize for faster processing while preserving aspect ratio
        orig_h, orig_w = img.shape[:2]
        scale = 500.0 / orig_w
        resized = cv2.resize(img, None, fx=scale, fy=scale)

        # Convert to grayscale and blur
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edged = cv2.Canny(blurred, 75, 200)

        if self.debug:
            cv2.imwrite(str(self.debug_dir / "01_edged.jpg"), edged)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        min_area_ratio = 0.20  # Ignore tiny quads (logos/boxes)
        img_area = resized.shape[0] * resized.shape[1]

        # Find contour with 4 vertices (quadrilateral)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (img_area * min_area_ratio):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Scale points back to original image size
                points = approx.reshape(4, 2) / scale
                return points.astype(np.float32)

        return None

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left (smallest sum)
        rect[2] = pts[np.argmax(s)]  # bottom-right (largest sum)

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right (smallest diff)
        rect[3] = pts[np.argmax(diff)]  # bottom-left (largest diff)
        return rect

    def perspective_transform(self, img: np.ndarray,
                             pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get top-down view of document."""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width and height of new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for top-down view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Compute and apply perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped

    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Auto-deskew using text line contour analysis.
        Based on OpenCV deskew techniques using minAreaRect.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        # Invert and threshold
        gray = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Dilate to connect text into lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel)

        # Find contours and extract angles
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        angles = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # Filter noise
                continue
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]  # Correct extraction: (center, size, angle)
            if angle != 90.0 and angle != -0.0:
                angles.append(angle)

        if not angles:
            return img

        # Use median angle for robustness
        angles.sort()
        angle = angles[len(angles) // 2]

        # Adjust angle range
        if angle > 45:
            angle = -(90 - angle)
        elif angle < -45:
            angle = 90 + angle

        # Rotate image
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

        return rotated

    def enhance_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Final preprocessing steps optimized for OCR."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if too small (Tesseract works better with larger text)
        h, w = gray.shape
        if h < 1000:
            scale = 1000.0 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        # Normalize local contrast (useful for phone-camera shadows)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # Denoise while preserving edges of receipt text
        denoised = cv2.bilateralFilter(contrast, 9, 75, 75)

        # Adaptive thresholding for uneven lighting
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 12
        )

        # Small morphology cleanup to strengthen text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.medianBlur(cleaned, 3)

        return cleaned

    def process(self, source, auto_detect: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            source: Image source (path, array, or camera index)
            auto_detect: Whether to auto-detect document boundaries

        Returns:
            Preprocessed image ready for OCR
        """
        # Load image
        if isinstance(source, int):
            img = self.capture_from_camera(source)
        else:
            img = self.load_image(source)

        if self.debug:
            cv2.imwrite(str(self.debug_dir / "00_original.jpg"), img)

        # Step 1: Detect and correct perspective
        if auto_detect:
            pts = self.detect_document_contour(img)
            if pts is not None:
                candidate = self.perspective_transform(img, pts)
                # Keep perspective output only when it is not an implausibly tiny crop.
                if candidate.shape[0] * candidate.shape[1] >= (img.shape[0] * img.shape[1] * 0.20):
                    img = candidate
                else:
                    print("Warning: Perspective crop too small. Using full image.")
                if self.debug:
                    cv2.imwrite(str(self.debug_dir / "02_perspective.jpg"), candidate)
            else:
                print("Warning: Could not detect document contour. Using full image.")

        # Step 2: Deskew
        img = self.deskew(img)
        if self.debug:
            cv2.imwrite(str(self.debug_dir / "03_deskewed.jpg"), img)

        # Step 3: Enhance for OCR
        processed = self.enhance_for_ocr(img)
        if self.debug:
            cv2.imwrite(str(self.debug_dir / "04_enhanced.jpg"), processed)

        return processed


class ReceiptOCR:
    """OCR extraction using Tesseract with receipt-specific configurations."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Tesseract config optimized for receipts
        self.configs = {
            'receipt': '--oem 3 --psm 6 -c preserve_interword_spaces=1 '
                       '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$%&()-:; ',
            'detailed': '--psm 4',
            'raw': '--psm 3'
        }

    def extract_text(self, img: np.ndarray, mode: str = 'receipt') -> str:
        """
        Extract text from preprocessed image.
        Runs multiple OCR passes and returns the most informative result.
        """
        base_config = self.configs.get(mode, self.configs['receipt'])
        passes = [
            (img, base_config),
            (img, '--oem 3 --psm 4 -c preserve_interword_spaces=1'),
            (img, '--oem 3 --psm 11 -c preserve_interword_spaces=1'),
            (cv2.bitwise_not(img), base_config),
        ]

        texts: List[str] = []
        for ocr_img, config in passes:
            try:
                text = pytesseract.image_to_string(ocr_img, config=config)
                texts.append(text)
            except Exception:
                continue

        if not texts:
            return ""

        def score(candidate: str) -> int:
            lines = [ln.strip() for ln in candidate.split('\n') if ln.strip()]
            digits = sum(ch.isdigit() for ch in candidate)
            keywords = ['TOTAL', 'BELANJA', 'INDOMARET', 'TUNAI', 'PPN']
            keyword_hits = sum(1 for key in keywords if key in candidate.upper())
            return len(lines) + digits + (5 * keyword_hits)

        return max(texts, key=score)

    def _normalize_ocr_text(self, text: str) -> str:
        """Normalize common OCR artifacts for Indonesian receipt parsing."""
        replacements = {
            '—': '-',
            '_': '-',
            '“': '"',
            '”': '"',
            '|': '1',
        }
        out = text
        for old, new in replacements.items():
            out = out.replace(old, new)
        return out

    def _parse_idr_amount(self, token: str) -> Optional[str]:
        """Parse and normalize amount token into canonical integer-like IDR string."""
        if not token:
            return None
        cleaned = re.sub(r'[^0-9.,]', '', token)
        if not cleaned:
            return None
        digits = re.sub(r'[^0-9]', '', cleaned)
        if not digits:
            return None
        return digits

    def _extract_location(self, lines: List[str]) -> Optional[str]:
        """
        Extract top location block (before first separator/date).
        This follows common Indomaret receipt structure.
        """
        location_lines = []
        stop_tokens = ('TOTAL', 'BELANJA', 'PPN', 'QR', 'TRXID')
        for line in lines[:12]:
            if re.search(r'\d{2}[./-]\d{2}[./-]\d{2,4}', line):
                break
            if re.match(r'^[-=]{4,}$', line):
                break
            if any(token in line.upper() for token in stop_tokens):
                break
            if len(line) >= 4:
                location_lines.append(line)
        if location_lines:
            return " | ".join(location_lines[:4])
        return None

    def _extract_date_time(self, text: str, lines: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract date and time from common transaction strings such as:
        06.05.26-15:20/...
        19 04.26-17:32/...
        """
        # 1) Prefer line-level parsing where time is present.
        for line in lines:
            if re.search(r'\d{1,2}[:.]\d{2}', line):
                # Any date-like token before time.
                dt = re.search(
                    r'(\d{1,2}\s+\d{1,2}[./-]\d{2,4}|\d{1,2}(?:[./-]\d{1,2}){1,2})\s*[- ]\s*(\d{1,2}[:.]\d{2})',
                    line
                )
                if dt:
                    raw_date = dt.group(1)
                    time_val = dt.group(2).replace('.', ':')
                    # Normalize DD MM.YY -> DD.MM.YY style
                    if re.match(r'^\d{1,2}\s+\d{1,2}[./-]\d{2,4}$', raw_date):
                        raw_date = raw_date.replace(' ', '.')
                    date_val = raw_date.replace('.', '/').replace('-', '/')
                    return date_val, time_val

                # Fallback: find a date token anywhere on the same line.
                date_token = re.search(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', line)
                time_token = re.search(r'\b\d{1,2}[:.]\d{2}\b', line)
                if date_token and time_token:
                    return (
                        date_token.group(0).replace('.', '/').replace('-', '/'),
                        time_token.group(0).replace('.', ':')
                    )

        # 2) Global fallback.
        dt_match = re.search(
            r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*[- ]\s*(\d{1,2}[:.]\d{2})',
            text
        )
        if dt_match:
            return (
                dt_match.group(1).replace('.', '/').replace('-', '/'),
                dt_match.group(2).replace('.', ':')
            )
        return None, None

    def _extract_items_indomaret(self, lines: List[str]) -> List[Dict]:
        """Extract line-items using Indonesian minimarket row patterns."""
        items: List[Dict] = []
        ignore_keywords = (
            'TOTAL', 'BELANJA', 'TUNAI', 'NON TUNAI', 'KEMBALI', 'PPN', 'DPP',
            'HARGA JUAL', 'LAYANAN', 'KONTAK', 'SMS/WA', 'TRXID', 'CANCEL'
        )

        # Pattern: <name> <qty> <unit_price> <line_total>
        pattern_three_numbers = re.compile(
            r'^(?P<name>.+?)\s+(?P<qty>\d{1,3})\s+(?P<unit>[0-9][0-9.,]{2,})\s+(?P<total>[0-9][0-9.,]{2,})$'
        )
        # Pattern: <name> <line_total> (fallback)
        pattern_single_total = re.compile(
            r'^(?P<name>.+?)\s+(?P<total>[0-9][0-9.,]{2,})$'
        )

        def is_plausible_item_name(name: str) -> bool:
            name = name.strip()
            if len(name) < 3 or len(name) > 40:
                return False
            if ':' in name:
                return False
            letters = [c for c in name if c.isalpha()]
            if not letters:
                return False
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio < 0.65:
                return False
            return True

        # Parse only inside body block: after transaction datetime and before total block.
        start_idx = 0
        end_idx = len(lines)
        dt_line_pattern = re.compile(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*[- ]\s*\d{1,2}[:.]\d{2}')
        for i, line in enumerate(lines):
            if dt_line_pattern.search(line):
                start_idx = i + 1
                break
        for i in range(start_idx, len(lines)):
            if 'TOTAL BELANJA' in lines[i].upper() or re.match(r'^TOTAL\b', lines[i].upper()):
                end_idx = i
                break

        for line in lines[start_idx:end_idx]:
            upper = line.upper()
            if any(keyword in upper for keyword in ignore_keywords):
                continue
            if len(line) < 6:
                continue

            m3 = pattern_three_numbers.match(line)
            if m3:
                name = m3.group('name').strip()
                qty = m3.group('qty')
                unit_price = self._parse_idr_amount(m3.group('unit'))
                total_price = self._parse_idr_amount(m3.group('total'))
                if (
                    name and total_price and
                    int(total_price) >= 1000 and
                    is_plausible_item_name(name)
                ):
                    item = {
                        'description': name,
                        'quantity': int(qty),
                        'unit_price': unit_price,
                        'price': total_price
                    }
                    items.append(item)
                continue

            m1 = pattern_single_total.match(line)
            if m1:
                name = m1.group('name').strip()
                total_price = self._parse_idr_amount(m1.group('total'))
                if (
                    name and total_price and
                    int(total_price) >= 1000 and
                    is_plausible_item_name(name) and
                    not re.search(r'\d{3,}\s*$', name)
                ):
                    items.append({'description': name, 'price': total_price})

        # Second pass: multi-line rows (name on one line, qty/price on next lines)
        amount_pat = re.compile(r'\b\d[\d.,]{2,}\b')
        idx = start_idx
        while idx < end_idx:
            name_line = lines[idx].strip()
            upper = name_line.upper()
            if (
                len(name_line) >= 5 and
                is_plausible_item_name(name_line) and
                not any(keyword in upper for keyword in ignore_keywords)
            ):
                window = " ".join(lines[idx + 1:min(idx + 4, end_idx)])
                amounts = amount_pat.findall(window)
                qty_match = re.search(r'\b([1-9]\d{0,2})\b', window)
                if amounts:
                    total_price = self._parse_idr_amount(amounts[-1])
                    unit_price = self._parse_idr_amount(amounts[-2]) if len(amounts) >= 2 else total_price
                    if total_price and int(total_price) >= 1000:
                        item = {
                            'description': re.sub(r'^[^A-Za-z0-9/]+', '', re.sub(r'\s+\d[\d.,]{2,}$', '', name_line)).strip(),
                            'price': total_price
                        }
                        if not is_plausible_item_name(item['description']):
                            idx += 1
                            continue
                        if qty_match:
                            item['quantity'] = int(qty_match.group(1))
                        if unit_price:
                            item['unit_price'] = unit_price
                        if not any(existing.get('description') == item['description'] and
                                   existing.get('price') == item['price'] for existing in items):
                            items.append(item)
                        idx += 3
                        continue
            idx += 1

        return items

    def extract_data(self, text: str) -> Dict:
        """
        Parse OCR text into structured receipt data.
        Tuned for Indonesian minimarket receipts (e.g., Indomaret).
        """
        text = self._normalize_ocr_text(text)
        data = {
            'raw_text': text,
            'vendor': None,
            'location': None,
            'date': None,
            'time': None,
            'total': None,
            'tax': None,
            'items': [],
            'payment_method': None,
            'confidence': 'low'
        }

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Vendor + location
        upper_text = text.upper()
        if re.search(r'INDOMAR|KLIKINDOM|INDO', upper_text):
            data['vendor'] = 'Indomaret'
        data['location'] = self._extract_location(lines)
        if not data['vendor'] and lines:
            first = lines[0]
            if len(first) >= 3 and re.search(r'[A-Za-z]', first):
                data['vendor'] = first

        # Extract Indonesian date/time patterns, e.g. 06.05.26-15:20/...
        parsed_date, parsed_time = self._extract_date_time(text, lines)
        if parsed_date:
            data['date'] = parsed_date
        if parsed_time:
            data['time'] = parsed_time

        # Fallback date extraction
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{1,2}[.]\d{1,2}[.]\d{2,4})\b',
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
        ]
        if not data['date']:
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data['date'] = match.group(1).replace('.', '/').replace('-', '/')
                    break

        # Extract total amount with Indonesian keywords
        total_patterns = [
            r'(?:TOTAL\s*BELANJA|TOTAL)[^\d]*(\d[\d.,]{2,})',
            r'(?:NON\s*TUNAI|TUNAI)[^\d]*(\d[\d.,]{2,})',
            r'(?:AMOUNT DUE|Grand Total)[^\d]*(\d[\d.,]{2,})',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                parsed = self._parse_idr_amount(match.group(1))
                if parsed:
                    data['total'] = parsed
                break

        # Extract tax
        tax_match = re.search(r'(?:PPN|TAX|GST|VAT)[^\d]*(\d[\d.,]{2,})',
                              text, re.IGNORECASE)
        if tax_match:
            data['tax'] = self._parse_idr_amount(tax_match.group(1))

        # Extract items using receipt-row patterns
        data['items'] = self._extract_items_indomaret(lines)

        # Payment method
        payment_keywords = ['NON TUNAI', 'TUNAI', 'DEBIT', 'CREDIT', 'QR', 'VISA',
                            'MASTERCARD', 'AMEX', 'CHECK', 'GIFT CARD']
        for keyword in payment_keywords:
            if keyword in text.upper():
                data['payment_method'] = keyword.title()
                break

        # Confidence scoring
        score = 0
        if data['date']: score += 1
        if data['total']: score += 1
        if data['vendor']: score += 1
        if len(data['items']) > 0: score += 1

        if score >= 3:
            data['confidence'] = 'high'
        elif score >= 2:
            data['confidence'] = 'medium'

        return data


class ReceiptDatabase:
    """SQLite-based storage for receipt data with search and export."""

    def __init__(self, db_path: str = "receipts.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    vendor TEXT,
                    date TEXT,
                    total REAL,
                    tax REAL,
                    payment_method TEXT,
                    confidence TEXT,
                    raw_text TEXT,
                    image_path TEXT,
                    items_json TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS receipt_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id INTEGER,
                    description TEXT,
                    price REAL,
                    FOREIGN KEY (receipt_id) REFERENCES receipts(id)
                )
            ''')

            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_vendor ON receipts(vendor)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_date ON receipts(date)
            ''')
            conn.commit()

    def insert(self, data: Dict, image_path: Optional[str] = None) -> int:
        """Insert receipt data and return receipt ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO receipts
                (vendor, date, total, tax, payment_method, confidence,
                 raw_text, image_path, items_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('vendor'),
                data.get('date'),
                self._parse_amount(data.get('total')),
                self._parse_amount(data.get('tax')),
                data.get('payment_method'),
                data.get('confidence'),
                data.get('raw_text'),
                image_path,
                json.dumps(data.get('items', []))
            ))

            receipt_id = cursor.lastrowid

            # Insert individual items
            for item in data.get('items', []):
                conn.execute('''
                    INSERT INTO receipt_items (receipt_id, description, price)
                    VALUES (?, ?, ?)
                ''', (receipt_id, item.get('description'),
                      self._parse_amount(item.get('price'))))

            conn.commit()
            return receipt_id

    def _parse_amount(self, amount_str) -> Optional[float]:
        """Parse amount string to float."""
        if amount_str is None:
            return None
        try:
            # Handle both 1,234.56 and 1.234,56 formats
            cleaned = str(amount_str).replace(',', '.')
            # If multiple dots, keep only last
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def search(self, vendor: Optional[str] = None,
               date_from: Optional[str] = None,
               date_to: Optional[str] = None,
               min_amount: Optional[float] = None,
               max_amount: Optional[float] = None) -> List[Dict]:
        """Search receipts with filters."""
        query = "SELECT * FROM receipts WHERE 1=1"
        params = []

        if vendor:
            query += " AND vendor LIKE ?"
            params.append(f"%{vendor}%")
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND date <= ?"
            params.append(date_to)
        if min_amount is not None:
            query += " AND total >= ?"
            params.append(min_amount)
        if max_amount is not None:
            query += " AND total <= ?"
            params.append(max_amount)

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def export_csv(self, filepath: str, **filters):
        """Export search results to CSV."""
        receipts = self.search(**filters)

        if not receipts:
            print("No receipts found for export.")
            return

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=receipts[0].keys())
            writer.writeheader()
            writer.writerows(receipts)

        print(f"Exported {len(receipts)} receipts to {filepath}")

    def get_stats(self) -> Dict:
        """Get spending statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT
                    COUNT(*) as total_receipts,
                    COALESCE(SUM(total), 0) as total_spent,
                    COALESCE(AVG(total), 0) as avg_receipt,
                    MAX(total) as max_receipt,
                    MIN(total) as min_receipt
                FROM receipts
                WHERE total IS NOT NULL
            ''')
            return dict(cursor.fetchone())


class ReceiptScanner:
    """Main orchestrator class combining all components."""

    def __init__(self, debug: bool = False, db_path: str = "receipts.db"):
        self.preprocessor = ReceiptPreprocessor(debug=debug)
        self.ocr = ReceiptOCR() if TESSERACT_AVAILABLE else None
        self.db = ReceiptDatabase(db_path)
        self.debug = debug

    def _open_camera_stream(self, camera_id: int = 0) -> cv2.VideoCapture:
        """Open camera stream with backend fallbacks."""
        backends = [None]
        if os.name == "nt":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        last_error = "Unknown camera stream error"
        for backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(camera_id)
            else:
                cap = cv2.VideoCapture(camera_id, backend)

            if cap.isOpened():
                return cap
            cap.release()
            last_error = f"Could not open camera stream {camera_id} (backend={backend})"

        raise RuntimeError(last_error)

    def _evaluate_capture_quality(
        self,
        frame: np.ndarray,
        prev_gray: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate frame quality for live auto-capture.
        Returns metrics, booleans, and optional contour points.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        frame_area = float(h * w)

        pts = self.preprocessor.detect_document_contour(frame)
        contour_ok = pts is not None
        area_ratio = 0.0
        aspect_ratio = 0.0
        if pts is not None:
            rect = self.preprocessor.order_points(pts)
            (tl, tr, br, bl) = rect
            width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
            height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
            if width > 0 and height > 0:
                area_ratio = float((width * height) / frame_area)
                aspect_ratio = float(max(width, height) / min(width, height))

        brightness = float(np.mean(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if prev_gray is None:
            motion = 0.0
        else:
            motion = float(np.mean(cv2.absdiff(gray, prev_gray)))

        checks = {
            "contour": contour_ok,
            "size": area_ratio >= 0.20,
            "aspect": 1.4 <= aspect_ratio <= 4.5 if contour_ok else False,
            "focus": blur_var >= 80.0,
            "exposure": 70.0 <= brightness <= 205.0,
            "steady": motion <= 6.0 if prev_gray is not None else True,
        }

        ready = all(checks.values())
        hint = "Ready - press Enter/Space"
        if not checks["contour"]:
            hint = "Find receipt edges"
        elif not checks["size"]:
            hint = "Move closer"
        elif not checks["aspect"]:
            hint = "Align receipt shape"
        elif not checks["focus"]:
            hint = "Hold still (blurry)"
        elif not checks["exposure"]:
            hint = "Adjust lighting"
        elif not checks["steady"]:
            hint = "Hold steady"

        return {
            "pts": pts,
            "checks": checks,
            "ready": ready,
            "hint": hint,
            "brightness": brightness,
            "blur_var": blur_var,
            "motion": motion,
        }

    def capture_live_receipt(
        self,
        camera_id: int = 0,
        auto_capture: bool = False,
        stable_frames: int = 12,
        show_scanner_preview: bool = True
    ) -> Optional[np.ndarray]:
        """
        Live receipt capture with visual guidance.
        - Enter/Space: manual capture
        - Q/Esc: cancel
        - Auto mode captures after stable 'ready' frames
        """
        cap = self._open_camera_stream(camera_id)
        prev_gray = None
        stable_count = 0
        captured = None
        countdown_started_at = None
        countdown_seconds = 1.5

        print("Live view started.")
        print("Controls: Enter/Space capture | Q/Esc cancel")
        if auto_capture:
            print("Auto-capture enabled.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                quality = self._evaluate_capture_quality(frame, prev_gray)
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if quality["ready"]:
                    stable_count += 1
                else:
                    stable_count = 0
                    countdown_started_at = None

                preview = frame.copy()
                pts = quality["pts"]
                if pts is not None:
                    poly = pts.reshape((-1, 1, 2)).astype(np.int32)
                    color = (0, 255, 0) if quality["ready"] else (0, 200, 255)
                    cv2.polylines(preview, [poly], True, color, 2)

                status_color = (0, 255, 0) if quality["ready"] else (0, 200, 255)
                cv2.putText(preview, quality["hint"], (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
                cv2.putText(preview,
                            f"Focus:{quality['blur_var']:.0f} Light:{quality['brightness']:.0f} Motion:{quality['motion']:.1f}",
                            (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(preview,
                            f"Stable: {stable_count}/{stable_frames}",
                            (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                if auto_capture and stable_count >= stable_frames:
                    if countdown_started_at is None:
                        countdown_started_at = time.time()
                    elapsed = time.time() - countdown_started_at
                    remaining = max(0.0, countdown_seconds - elapsed)
                    cv2.putText(preview, f"Auto capture in {remaining:.1f}s",
                                (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    if elapsed >= countdown_seconds:
                        captured = frame.copy()
                        break

                if show_scanner_preview:
                    if pts is not None:
                        candidate = self.preprocessor.perspective_transform(frame, pts)
                    else:
                        candidate = frame
                    try:
                        deskewed = self.preprocessor.deskew(candidate)
                        enhanced = self.preprocessor.enhance_for_ocr(deskewed)
                        cv2.imshow("Scanner Preview", enhanced)
                    except Exception:
                        cv2.imshow("Scanner Preview", candidate)

                cv2.imshow("Camera Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 32):  # Enter / Space
                    captured = frame.copy()
                    break
                if key in (27, ord('q'), ord('Q')):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return captured

    def scan(self, source, auto_detect: bool = True,
             save_image: bool = True) -> Dict:
        """
        Scan a receipt from source and store in database.

        Args:
            source: Image path, camera index (int), or numpy array
            auto_detect: Auto-detect document boundaries
            save_image: Save processed image to disk

        Returns:
            Extracted receipt data dictionary
        """
        # Generate unique filename for this scan
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = None

        # Step 1: Preprocess
        print("Step 1: Preprocessing image...")
        processed_img = self.preprocessor.process(source, auto_detect=auto_detect)

        if save_image:
            image_dir = Path("scanned_receipts")
            image_dir.mkdir(exist_ok=True)
            image_path = str(image_dir / f"receipt_{timestamp}.jpg")
            cv2.imwrite(image_path, processed_img)
            print(f"Saved processed image to {image_path}")

        # Step 2: OCR
        if self.ocr is None:
            print("WARNING: OCR not available. Returning raw image only.")
            return {'raw_text': '', 'image_path': image_path}

        print("Step 2: Running OCR...")
        raw_text = self.ocr.extract_text(processed_img)

        if self.debug:
            print("\n--- RAW OCR TEXT ---")
            print(raw_text)
            print("--- END RAW TEXT ---\n")

        # Step 3: Parse structured data
        print("Step 3: Parsing receipt data...")
        data = self.ocr.extract_data(raw_text)
        data['image_path'] = image_path

        # Step 4: Store in database
        print("Step 4: Saving to database...")
        receipt_id = self.db.insert(data, image_path)
        data['receipt_id'] = receipt_id

        print(f"[OK] Receipt saved with ID: {receipt_id}")
        print(f"  Vendor: {data.get('vendor', 'Unknown')}")
        print(f"  Date: {data.get('date', 'Unknown')}")
        print(f"  Total: {data.get('total', 'Unknown')}")
        print(f"  Confidence: {data.get('confidence', 'low')}")

        return data

    def scan_batch(self, folder_path: str, pattern: Optional[str] = None) -> List[Dict]:
        """Process all images in a folder."""
        folder = Path(folder_path)
        results = []

        patterns = [pattern] if pattern else ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        image_paths = []
        for p in patterns:
            image_paths.extend(folder.glob(p))

        # Deduplicate while keeping stable order
        seen = set()
        unique_paths = []
        for img_path in image_paths:
            key = str(img_path.resolve())
            if key not in seen:
                seen.add(key)
                unique_paths.append(img_path)

        for img_path in unique_paths:
            print(f"\nProcessing: {img_path.name}")
            try:
                result = self.scan(img_path)
                results.append(result)
            except Exception as e:
                print(f"ERROR processing {img_path}: {e}")

        print(f"\nBatch complete: {len(results)} receipts processed")
        return results

    def query(self, **filters) -> List[Dict]:
        """Query stored receipts."""
        return self.db.search(**filters)

    def export(self, filepath: str, **filters):
        """Export receipts to CSV."""
        self.db.export_csv(filepath, **filters)

    def stats(self) -> Dict:
        """Get spending statistics."""
        return self.db.get_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Treasurer Receipt Scanner / Logger"
    )
    parser.add_argument("source", nargs="?", help="Image path or camera index")
    parser.add_argument("--camera", "-c", action="store_true",
                       help="Use camera for capture")
    parser.add_argument("--live-preview", action="store_true",
                       help="Show live camera/scanner preview before capture")
    parser.add_argument("--auto-capture", action="store_true",
                       help="Auto-capture when receipt is aligned and stable")
    parser.add_argument("--stable-frames", type=int, default=12,
                       help="Stable ready frames required before auto-capture")
    parser.add_argument("--no-scanner-preview", action="store_true",
                       help="Disable second scanner preview window in live mode")
    parser.add_argument("--batch", "-b", help="Process all images in folder")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode with intermediate images")
    parser.add_argument("--query", "-q", help="Search vendor name")
    parser.add_argument("--export", "-e", help="Export to CSV file")
    parser.add_argument("--stats", "-s", action="store_true",
                       help="Show spending statistics")
    parser.add_argument("--db", default="receipts.db",
                       help="Database file path")

    args = parser.parse_args()

    scanner = ReceiptScanner(debug=args.debug, db_path=args.db)

    if args.stats:
        stats = scanner.stats()
        print("\n=== SPENDING STATISTICS ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.query:
        results = scanner.query(vendor=args.query)
        print(f"\nFound {len(results)} receipts matching '{args.query}':")
        for r in results:
            print(f"  [{r['id']}] {r['date']} | {r['vendor']} | ${r['total']}")

    elif args.export:
        scanner.export(args.export)

    elif args.batch:
        scanner.scan_batch(args.batch)

    elif args.camera or (args.source and args.source.isdigit()):
        cam_id = int(args.source) if args.source else 0
        if args.live_preview or args.auto_capture:
            frame = scanner.capture_live_receipt(
                camera_id=cam_id,
                auto_capture=args.auto_capture,
                stable_frames=max(1, args.stable_frames),
                show_scanner_preview=not args.no_scanner_preview
            )
            if frame is None:
                print("Capture canceled.")
                return
            scanner.scan(frame)
        else:
            print(f"Capturing from camera {cam_id}...")
            print("Press Enter when ready (or Ctrl+C to cancel)")
            input()
            scanner.scan(cam_id)

    elif args.source:
        scanner.scan(args.source)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()