from tkinter.ttk import setup_master
from typing import List, Dict
from sqlalchemy.orm import Session
from pdf2image import convert_from_path
import pytesseract

from models.champs import Champs


class Recognition_service:
    def __init__(self, pdf_path, idtypelivrable, db ,tesseract_cmd="/usr/bin/tesseract", dpi=300):
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
        self.temp_image_path = "temp_page.png"
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.selected_boxes = []
        self.db = db

    def extract_text_from_box(self, box, page):
        """
        Extrait le texte contenu dans les coordonnées spécifiées sur la page indiquée.

        :param box: Coordonnées (x1, y1, x2, y2) de la zone à extraire.
        :param page: Numéro de la page (commençant à 1).
        :return: Texte extrait de la zone.
        """
        # Ajuste l'index de la page (les pages sont indexées à partir de 0)
        cropped = self.pages[page - 1].crop(box)
        return pytesseract.image_to_string(cropped)

    def process_selected_boxe(self, champ):
        """
        Extrait le texte de la zone et de la page indiquées pour un objet 'champ'.

        :param champ: Objet Champs contenant les informations de zone et page.
        :return: Texte extrait de la zone spécifiée.
        """
        return self.extract_text_from_box(champ.zone, champ.page)

    def process_champs_list(self, champs_list: List['Champs']) -> Dict[str, str]:
        """
        Parcourt une liste d'objets Champs, extrait le texte de leur zone et page,
        et renvoie un dictionnaire avec le nom du champ ('nom') comme clé et le texte extrait comme valeur.

        :param champs_list: Liste d'objets Champs.
        :return: Dictionnaire {nom_du_champ: texte_extrait}.
        """
        print(champs_list)
        results = {}
        # for champ in champs_list:
        #     extracted_text = self.process_selected_boxe(champ)
        #     results[champ.nom] = extracted_text
        return results

    def get_champs_list(self) -> List['Champs']:
        """
        Récupère la liste des objets Champs depuis la base de données MySQL,
        filtrée par l'idtypelivrable fourni lors de l'initialisation.

        :param session: Session SQLAlchemy active.
        :return: Liste d'objets Champs correspondant au filtre.
        """
        champs_list = self.db.session.query(Champs).filter_by(type_livrable_id=self.typelivrable).all()
        return champs_list

    def process(self) -> Dict[str, str]:
        """
        Recupère la list des champs execute l'OCR pour retourner un dictionnaire
        :param session:
        :return:
        """
        return self.process_champs_list(self.get_champs_list())
