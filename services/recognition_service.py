import cv2
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pytesseract

class recognition_service:
    def __init__(self, pdf_path, tesseract_cmd="/usr/bin/tesseract", dpi=300):
        self.pdf_path = pdf_path
        self.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self.pages = convert_from_path(pdf_path, dpi=dpi)
        self.current_page = self.pages[0]
        self.temp_image_path = "temp_page.png"
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.selected_boxes = []

    def save_temp_image(self):
        """Save the current page as a temporary image."""
        self.current_page.save(self.temp_image_path)

    def extract_text_from_box(self, box):
        """Extract text from a specific box in the image."""
        cropped = self.current_page.crop(box)
        return pytesseract.image_to_string(cropped)

    def ocr_with_boxes(self, keyword):
        """Perform OCR and locate bounding boxes for a specific keyword."""
        ocr_data = pytesseract.image_to_data(self.current_page, output_type=pytesseract.Output.DICT)
        found_boxes = []

        for i, word in enumerate(ocr_data['text']):
            if keyword.lower() in word.lower():
                x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                              ocr_data['width'][i], ocr_data['height'][i])
                found_boxes.append((x, y, x + w, y + h))
        return found_boxes
