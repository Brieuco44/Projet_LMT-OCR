from pdf2image import convert_from_path
import pytesseract

class recognition_service:
    def __init__(self, pdf_path, tesseract_cmd="/usr/bin/tesseract", dpi=300):
        self.pdf_path = pdf_path
        self.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self.pages = convert_from_path(pdf_path, dpi=dpi)
        self.temp_image_path = "temp_page.png"
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.selected_boxes = []

    def extract_text_from_box(self, box, page):
        """Extrait le text contenu aux coordonnées"""
        cropped = self.pages[page-1].crop(box) # utilisation de la page moins 1
        return pytesseract.image_to_string(cropped)

    def process_selected_boxes(self, selected_boxes):
        """Parcours la liste des zones à extraire le texte"""
        for i, Champs in enumerate(selected_boxes):
            text = self.extract_text_from_box(Champs.zone,Champs.page)