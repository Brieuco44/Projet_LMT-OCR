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