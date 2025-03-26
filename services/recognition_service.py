from typing import List, Dict, Any
import json
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch
from models.champs import Champs
from models.zone import Zone
import re


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

        # Extrait le texte de la zone découpée
        return pytesseract.image_to_string(cropped, config=r'--oem 3 --psm 6').strip()

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
                if champ.question:
                    results[zone.libelle][champ.nom] = self.get_answer_from_text(extracted_element, champ.question)

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

    def get_answer_from_text(self, text, question):
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

