from pdf2image import convert_from_path
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

    def extract_text_from_box(self, box):
        """Extrait le text contenu aux coordonnées"""
        cropped = self.current_page.crop(box)
        return pytesseract.image_to_string(cropped)

    def process_selected_boxes(self, selected_boxes):
        """Parcours la liste des zones à extraire le texte"""
        for i, box in enumerate(selected_boxes):
            text = self.extract_text_from_box(box)