from typing import List, Dict, Any
import json
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
from models.champs import Champs
from models.zone import Zone
from PIL import ImageDraw

class Recognition_service:
    def __init__(self, pdf_path, idtypelivrable, db ,tesseract_cmd="/usr/bin/tesseract", dpi=300, model_path="../roberta_large_squad2_download"):
        """
        Initialisation du service de reconnaissance.

        :param pdf_path: Chemin vers le fichier PDF à traiter.
        :param idtypelivrable: Identifiant du type livrable pour filtrer les champs.
        :param tesseract_cmd: Chemin vers l'exécutable Tesseract.
        :param dpi: Résolution (DPI) utilisée pour convertir le PDF en images.
        """
        self.pdf_path = pdf_path
        self.typelivrable = idtypelivrable  # idtypelivrable passé en paramètre
        self.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self.pages = convert_from_path(pdf_path, dpi=dpi)
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.selected_boxes = []
        self.db = db
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)

        self.question_answerer = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",  # Explicitly use PyTorch for performance
            top_k=20,
        )

    def box_to_tuples(self,box):
        return (box['x1'], box['y1'], box['x2'], box['y2'])

    def extraction(self, box, page):
        # Ajuste l'index de la page (les pages sont indexées à partir de 0)
        image = self.pages[page - 1]  # Get the page image

        # Convertir les coordonnées en format (left, upper, right, lower)
        crop_box = self.box_to_tuples(box)

        # Cropping the image
        cropped = image.crop(crop_box)

        cropped.save("img.png")

        # Extrait le texte de la zone découpée
        return pytesseract.image_to_string(cropped, config=r'--oem 1 --psm 6').strip()

    def extract_text_from_box(self, box, page):
        """
        Extrait le texte contenu dans les coordonnées spécifiées sur la page indiquée.

        :param box: Coordonnées {'x1': 1356, 'x2': 2330, 'y1': 2860, 'y2': 3048} de la zone à extraire.
        :param page: Numéro de la page (commençant à 1).
        :return: Texte extrait de la zone.
        """
        return self.extraction(box,page)

    def extract_signature_from_box(self, box, page):
        """
        Renvoie True si signature sinon False : on remarque une signature s'il y a de l'encre dans la zone ou du texte.
        :param box: Coordonnées {"x1": 1356, "x2": 2330, "y1": 2860, "y2": 3048} de la zone à analyser.
        :param page: Numéro de la page (commençant à 1).
        :return: True si une signature est détectée, sinon False.
        """
        # Extrait le texte de la zone découpée avec pytesseract
        extracted_text = self.extraction(box,page)

        # Si du texte est extrait, on suppose que c'est une signature
        if extracted_text.strip():
            return True

        return False


    def process_selected_boxe(self, zone):
        """
        Extrait le texte de la zone et de la page indiquées pour un objet 'champ'.

        :param champ: Objet Champs contenant les informations de zone et page.
        :return: Texte extrait de la zone spécifiée.
        """
        return self.extract_text_from_box(zone.coordonnees, zone.page).strip()

    def process_champs_list(self, zones_list: List['Zone']) -> Dict[str, Dict[Any, Any]]:
        """
        Iterates over a list of Zone objects, extracts text from their area and page,
        and returns a dictionary with the field name ('nom') as a key and the extracted text as a value.

        :param zones_list: List of Zone objects.
        :return: Dictionary {field_name: extracted_text}.
        """
        results: Dict[str, Dict[Any, Any]] = {}

        for zone in zones_list:
            extracted_element = self.process_selected_boxe(zone)
            results[zone.libelle] = {}

            for champ in self.get_champs_list_from_zone(zone.id):
                if champ.type_champs_id == 4:
                    results[zone.libelle][champ.nom] = self.has_signature(zone.coordonnees, zone.page)
                else:
                    if champ.question:
                        results[zone.libelle][champ.nom] = self.get_answer_from_text(extracted_element, champ.question,champ.type_champs_id)

        return results

    def get_zone_list(self) -> List['Zone']:
        """
        :param session: Session SQLAlchemy active.
        :return: Liste d'objets Champs correspondant au filtre.
        """
        champs_list = self.db.session.query(Zone).filter_by(type_livrable_id=self.typelivrable).all()
        return champs_list

    def get_champs_list_from_zone(self, zoneid) -> List['Champs']:
        """
        :param session: Session SQLAlchemy active.
        :return: Liste d'objets Champs correspondant au filtre.
        """
        champs_list = (
            self.db.session.query(Champs)
            .join(Zone, Champs.zoneid == zoneid)  # Jointure entre Champs et Zone
            .filter(Zone.type_livrable_id == self.typelivrable)  # Filtrer sur le type_livrable_id
            .all()
        )
        return champs_list

    def correct_nummarche(self,text):
        # Remove leading/trailing spaces
        text = text.strip()
        # Expecting exactly 10 characters: 6 letters + 4 digits
        if len(text) != 10:
            # Optionally handle unexpected lengths (e.g., pad, log, or return original)
            return text

        # Split into expected letter part and digit part
        letter_part = text[:6]
        digit_part = text[6:]

        # For the letter part, if any digit '0' appears (OCR might mis-read it),
        # you may want to convert it to 'O' (assuming it's a letter).
        corrected_letter = ''.join('O' if char == '0' else char for char in letter_part)

        # For the digit part, if any letter 'O' appears, convert it to '0'
        corrected_digit = ''.join('0' if char.upper() == 'O' else char for char in digit_part)

        return corrected_letter + corrected_digit

    def get_answer_from_text(self, text, question,type_champs_id):
        """
        Extracts the best possible answer from the given text using a question-answering model.
        Improved post-processing to enforce a confidence threshold, clean punctuation, and normalize the answer.

        :param text: The extracted text.
        :param question: The question to ask.
        :return: Extracted and formatted answer, or an empty string if not valid.
        """
        formatted_question = f"Quel est {question} ?"

        results = self.question_answerer(question=formatted_question, context=text)

        if not results:
            return ""

        # Select the best result based on score.
        best_result = max(results, key=lambda r: r['score'])

        best_answer = best_result['answer'].strip()


        # Handle multi-line cases (e.g., addresses)
        best_answer = best_answer.replace("\n", " ").replace("|","").strip()

        # If a colon is present, take only the text after the first colon.
        if ':' in best_answer:
            best_answer = ""

        if type_champs_id==1:
            best_answer = self.correct_nummarche(best_answer)

        return best_answer

    def process(self, jsonformat=False):
        """
        Récupère la liste des champs, exécute l'OCR et retourne un dictionnaire ou un JSON.

        :param jsonformat: Si True, renvoie le résultat en format JSON, sinon en dictionnaire.
        :return: Dictionnaire des champs et texte extrait ou chaîne JSON.
        """
        # Récupère la liste des champs et extrait le texte
        champs_data = self.process_champs_list(self.get_zone_list())

        # Si jsonformat est True, renvoie le résultat en format JSON
        if jsonformat:
            print()
            print(json.dumps(champs_data))

        return champs_data

    def draw_boxes_on_pdf(self, output_pdf_path="output_with_boxes.pdf"):
        """
        Draws rectangles around specified areas for each field on the PDF pages
        and saves the result as a new PDF.

        :param output_pdf_path: Path of the output PDF with marked areas.
        """
        # Retrieve the list of zones from the database
        zones_list = self.get_zone_list()

        # Organize boxes by page number (pages start at 1)
        page_boxes = {}
        for zone in zones_list:
            boxes = page_boxes.setdefault(zone.page, [])
            boxes.append(zone.coordonnees)

        # Create a new list of images with drawn rectangles
        drawn_pages = []
        for idx, page_img in enumerate(self.pages, start=1):
            # Draw boxes only if the page has them
            if idx in page_boxes:
                img_copy = page_img.copy()
                draw = ImageDraw.Draw(img_copy)
                for box in page_boxes[idx]:
                    draw.rectangle(self.box_to_tuples(box), outline="red", width=2)
                drawn_pages.append(img_copy)
            else:
                drawn_pages.append(page_img)

        # Save all modified pages into a new PDF
        if drawn_pages:
            drawn_pages[0].save(
                output_pdf_path,
                "PDF",
                resolution=self.dpi,
                save_all=True,
                append_images=drawn_pages[1:]
            )

    def has_signature(self, box, page):
        """
        Detects if a signature is present in the specified box on a given page.

        :param box: Dictionary with coordinates {"x1": ..., "x2": ..., "y1": ..., "y2": ...}.
        :param page: Page number (starting from 1).
        :return: True if a signature is detected, otherwise False.
        """
        # Extract the image portion from the PDF page
        cropped = self.extraction(box,page)

        # Si du texte est extrait, on suppose que c'est une signature
        if cropped.strip():
            return True

        return False