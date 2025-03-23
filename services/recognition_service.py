from typing import List, Dict
import re
import json
from PIL import ImageDraw, ImageEnhance, ImageFilter
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
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.selected_boxes = []
        self.db = db

    def box_to_tuples(self,box):
        return (box['x1'], box['y1'], box['x2'], box['y2'])

    def extract_text_from_box(self, box, page):
        """
        Extrait le texte contenu dans les coordonnées spécifiées sur la page indiquée.

        :param box: Coordonnées {'x1': 1356, 'x2': 2330, 'y1': 2860, 'y2': 3048} de la zone à extraire.
        :param page: Numéro de la page (commençant à 1).
        :return: Texte extrait de la zone.
        """
        # Ajuste l'index de la page (les pages sont indexées à partir de 0)
        image = self.pages[page - 1]  # Get the page image

        # Convertir les coordonnées en format (left, upper, right, lower)
        crop_box = self.box_to_tuples(box)

        # Cropping the image
        cropped = image.crop(crop_box)


        # Extrait le texte de la zone découpée
        return pytesseract.image_to_string(cropped)

    def extract_signature_from_box(self, box, page):
        """
        Renvoie True si signature sinon False : on remarque une signature s'il y a de l'encre dans la zone ou du texte.
        :param box: Coordonnées {'x1': 1356, 'x2': 2330, 'y1': 2860, 'y2': 3048} de la zone à analyser.
        :param page: Numéro de la page (commençant à 1).
        :return: True si une signature est détectée, sinon False.
        """
        # Ajuste l'index de la page (les pages sont indexées à partir de 0)
        image = self.pages[page - 1]  # Get the page image

        # Convertir les coordonnées en format (left, upper, right, lower)
        crop_box = self.box_to_tuples(box)

        # Cropping the image
        cropped = image.crop(crop_box)

        # Extrait le texte de la zone découpée avec pytesseract
        extracted_text = pytesseract.image_to_string(cropped)

        # Si du texte est extrait, on suppose que c'est une signature
        if extracted_text.strip():
            return True

        # Si aucun texte n'est trouvé, vérifier la couleur de l'encre (ici on recherche des différences de couleur)
        if self.check_color_difference(
                cropped):  # Utilisez la méthode check_color_difference pour vérifier les couleurs
            return True

        return False

    def check_color_difference(self, cropped_image):
        """
        Vérifie s'il y a une différence de couleur dans l'image pour détecter de l'encre.
        :param cropped_image: L'image découpée à analyser.
        :return: True si une différence de couleur est détectée, sinon False.
        """
        pixels = cropped_image.load()

        width, height = cropped_image.size
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                # Vérifier si la couleur n'est pas grise (exemple de détection d'encre)
                if r != g or g != b:  # Simple exemple : on vérifie si le pixel n'est pas gris
                    return True

        return False

    def extract_date_from_box(self, box, page):
        """
        Extrait la date manuscrite (au format 19/04/2024 ou 19/04/24) d'une zone spécifiée sur une page.
        Si la date n'est pas reconnue, retourne "unknown".

        :param box: Coordonnées de la zone à analyser, par exemple {'x1': 1356, 'x2': 2330, 'y1': 2860, 'y2': 3048}.
        :param page: Numéro de la page à partir de 1.
        :return: Date extraite sous forme de chaîne de caractères ou "unknown" si la date n'est pas reconnue.
        """
        image = self.pages[page - 1]

        # Recadrage de la zone d'intérêt
        crop_box = self.box_to_tuples(box)

        cropped = image.crop(crop_box)

        # save the image
        cropped.save(str(crop_box) + '.jpg')  # Save the image

        # Configurer pytesseract pour optimiser la reconnaissance des chiffres et des séparateurs
        custom_config = r'--oem 3 --psm 6'
        ocr_result = pytesseract.image_to_string(cropped, config=custom_config, lang='eng+fra')

        # Nettoyer le texte reconnu
        ocr_result = ocr_result.strip()

        # Expression régulière pour détecter un format de date "dd/mm/yyyy" ou "dd/mm/yy"
        date_pattern = r'(\b\d{1,2}/\d{1,2}/(\d{2}|\d{4})\b)'
        match = re.search(date_pattern, ocr_result)

        if match:
            # On retourne la première occurrence trouvée
            return match.group(1)
        else:
            # Si aucune date n'est reconnue, on retourne "unknown"
            return "unknown"

    def process_selected_boxe(self, champ):
        """
        Extrait le texte de la zone et de la page indiquées pour un objet 'champ'.

        :param champ: Objet Champs contenant les informations de zone et page.
        :return: Texte extrait de la zone spécifiée.
        """
        if champ.type_champs_id == 4: # Signature
            return self.extract_signature_from_box(champ.zone, champ.page)
        if champ.type_champs_id == 5: # Date
            return self.extract_date_from_box(champ.zone, champ.page)
        else:
            return self.extract_text_from_box(champ.zone, champ.page).replace("/", "").strip()

    def process_champs_list(self, champs_list: List['Champs']) -> Dict[str, str]:
        """
        Parcourt une liste d'objets Champs, extrait le texte de leur zone et page,
        et renvoie un dictionnaire avec le nom du champ ('nom') comme clé et le texte extrait comme valeur.

        :param champs_list: Liste d'objets Champs.
        :return: Dictionnaire {nom_du_champ: texte_extrait}.
        """
        results = {}
        for champ in champs_list:
            extracted_element = self.process_selected_boxe(champ)
            results[champ.nom] = extracted_element
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

    def process(self, jsonformat=False):
        """
        Récupère la liste des champs, exécute l'OCR et retourne un dictionnaire ou un JSON.

        :param jsonformat: Si True, renvoie le résultat en format JSON, sinon en dictionnaire.
        :return: Dictionnaire des champs et texte extrait ou chaîne JSON.
        """
        # Récupère la liste des champs et extrait le texte
        champs_data = self.process_champs_list(self.get_champs_list())

        # Si jsonformat est True, renvoie le résultat en format JSON
        if jsonformat:
            print(json.dumps(champs_data))

        return champs_data

    def draw_boxes_on_pdf(self, output_pdf_path="output_with_boxes.pdf"):
        """
        Dessine des rectangles autour des zones spécifiées pour chaque champ sur les pages du PDF
        et sauvegarde le résultat dans un nouveau fichier PDF.

        :param output_pdf_path: Chemin du PDF de sortie avec les zones encadrées.
        """
        # Récupère la liste des champs depuis la base de données
        champs_list = self.get_champs_list()

        # Organise les boîtes par numéro de page (les pages commencent à 1)
        page_boxes = {}
        for champ in champs_list:
            # On suppose que 'champ.zone' est un tuple (x1, y1, x2, y2) et 'champ.page' est le numéro de page (commençant à 1)
            boxes = page_boxes.setdefault(champ.page, [])
            boxes.append(champ.zone)

        # Crée une nouvelle liste d'images avec les rectangles dessinés
        drawn_pages = []
        for idx, page_img in enumerate(self.pages, start=1):
            # Si la page contient des boîtes, les dessiner
            if idx in page_boxes:
                # Crée une copie de l'image pour ne pas altérer l'image originale
                img_copy = page_img.copy()
                draw = ImageDraw.Draw(img_copy)
                for box in page_boxes[idx]:
                    # Dessine un rectangle rouge avec une épaisseur de 2 pixels
                    draw.rectangle(self.box_to_tuples(box), outline="red", width=2)
                drawn_pages.append(img_copy)
            else:
                drawn_pages.append(page_img)

        # Sauvegarde toutes les pages modifiées dans un nouveau PDF
        if drawn_pages:
            drawn_pages[0].save(
                output_pdf_path,
                "PDF",
                resolution=self.dpi,
                save_all=True,
                append_images=drawn_pages[1:]
            )
