from typing import List, Dict, Any
import json

import numpy as np
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch

from PIL import ImageDraw
import cv2

from ultralytics import YOLO
import supervision as sv

from services.DateOCRProcessor import DateOCRProcessor


class Recognition_service:
    def __init__(self, pdf, idtypelivrable, db ,tesseract_cmd="/usr/bin/tesseract", dpi=300, model_path="../roberta_base_squad2_download",signature_model="../signature_model.pt"):
        """
        Initialisation du service de reconnaissance.

        :param pdf_path: Chemin vers le fichier PDF à traiter.
        :param idtypelivrable: Identifiant du type livrable pour filtrer les champs.
        :param tesseract_cmd: Chemin vers l'exécutable Tesseract.
        :param dpi: Résolution (DPI) utilisée pour convertir le PDF en images.
        """
        self.pdf_path = pdf
        self.typelivrable = idtypelivrable
        self.tesseract_cmd = tesseract_cmd
        self.dpi = dpi
        self.signature_model = signature_model

        if isinstance(pdf, str):
            self.pages = convert_from_path(pdf, dpi=dpi)
            # page 1 size
            print(self.pages[0].size)
        else:
            self.pages = convert_from_bytes(pdf.read(), dpi=dpi)
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
            framework="pt",
            top_k=5,
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

        #cropped.save(f"{box}.png")

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
                elif champ.type_champs_id == 7:
                    find = self.has_handwrittenDate(zone.coordonnees, zone.page)
                    results[zone.libelle][champ.nom] = find[0]
                    results[zone.libelle]["Confiance"] = f"{find[1] * 100:.1f} %" if find[1] is not None else ""
                else:
                    if champ.question:
                        results[zone.libelle][champ.nom] = self.get_answer_from_text(extracted_element, champ.question,champ.type_champs_id)

        return results

    def get_zone_list(self) -> List['Zone']:
        """
        :param session: Session SQLAlchemy active.
        :return: Liste d'objets Champs correspondant au filtre.
        """
        from models.zone import Zone
        champs_list = self.db.session.query(Zone).filter_by(type_livrable_id=self.typelivrable).all()
        return champs_list

    def get_champs_list_from_zone(self, zoneid) -> List['Champs']:
        """
        :param session: Session SQLAlchemy active.
        :return: Liste d'objets Champs correspondant au filtre.
        """
        from models.champs import Champs
        from models.zone import Zone
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

        # print(text)
        # print('------------')
        # print(formatted_question)
        # print('------------')
        # print(best_result)
        # print('------------')

        # Handle multi-line cases (e.g., addresses)
        best_answer = best_answer.replace("\n", " ").replace("|","").strip()

        # If there's a colon, split and take the part after it
        if ':' in best_answer:
            best_answer = best_answer.split(':', 1)[1].strip()

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


    def has_signature(self, box, page):
        # Load the saved model
        loaded_model = YOLO(self.signature_model)

        # Crop the image from the page using the box coordinates
        # (Ensure that self.pages contains PIL images and box_to_tuples returns a tuple suitable for PIL.crop)
        pil_image = self.pages[page - 1].crop(self.box_to_tuples(box))

        # Convert the PIL image to a NumPy array for OpenCV (PIL image is in RGB)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use the loaded model for inference on the cropped image
        results = loaded_model(image)

        # Process results to create detections
        detections = sv.Detections.from_ultralytics(results[0])

        # # Annotate the image with detection boxes (for visualization if needed)
        # box_annotator = sv.BoxAnnotator()
        # annotated_image = box_annotator.annotate(scene=image, detections=detections)

        # Display the annotated image (optional)
        # cv2.imshow("Detections", annotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Check if any detections were made
        if len(detections) == 0:
            return False
        else:
            return True

    def has_handwrittenDate(self, box, page):

        pil_image = self.pages[page - 1].crop(self.box_to_tuples(box))

        # Convert the PIL image to a NumPy array for OpenCV (PIL image is in RGB)
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use the loaded model for inference on the cropped image
        return DateOCRProcessor().run(image)

    def draw_boxes_on_pdf(self, output_pdf_path="output_with_boxes.pdf"):
        """
        Draws rectangles around specified areas for each field on the PDF pages
        and saves the result as a new PDF.

        :param output_pdf_path: Path of the output PDF with marked areas.
        """
        # Retrieve the list of zones from the database
        zones_list = self.get_zone_list()

        # Organize boxes by page number (pages start at 1)  392, 230, 697, 388 {'x1': 392, 'x2': 230, 'y1': 697, 'y2': 388}
#        page_boxes = {1: [{'x1': 1269, 'x2': 2324, 'y1': 759, 'y2': 1551 }]}
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
