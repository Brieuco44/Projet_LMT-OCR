import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from PIL import Image
import torch
import time


class Recognition_service:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.question_answerer = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer,
                                          device=0 if torch.cuda.is_available() else -1)

    @staticmethod
    def preprocess_image(img):
        """Amélioration du prétraitement de l'image pour un OCR plus précis."""
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # Réduction du bruit
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(img_thresh)

    def extract_text_from_pdf(self, pdf_path):
        """Extraction du texte à partir d'un PDF en utilisant OCR."""
        images = convert_from_path(pdf_path, dpi=300)
        text = ''
        for img in images:
            extracted_text = pytesseract.image_to_string(img, lang="fra")
            text += extracted_text + "\n"

        return text

    def get_answer_from_text(self, text, question):
        """Découpe le texte et sélectionne la meilleure réponse trouvée dans les chunks."""
        chunks = self.chunk_text(text)
        best_answer = None
        best_score = 0.0  # Score initialisé à 0

        for chunk in chunks:
            if not chunk.strip():  # Ignore les chunks vides
                continue

            results = self.question_answerer(question=question, context=chunk,
                                             top_k=3)  # Prend les 3 meilleures réponses
            # Sélectionne la meilleure réponse du chunk
            best_local_result = max(results, key=lambda r: r['score'])  # Prend la réponse avec le score max
            # print(best_local_result)
            # Garde la meilleure réponse globale
            if best_local_result['score'] > best_score:
                best_answer = best_local_result['answer']
                best_score = best_local_result['score']

        return best_answer.strip() if best_answer else "Aucune réponse fiable trouvée"

    @staticmethod
    def chunk_text(text, max_words=1000, overlap=200):  # Augmenter la taille des chunks
        """Découpe le texte en morceaux de max_words mots, supprime les sauts de ligne et les met sur une seule ligne."""
        words = text.split()  # Découpe en mots
        chunks, chunk = [], []

        for word in words:
            chunk.append(word)
            if len(chunk) >= max_words:
                chunks.append(" ".join(chunk[:-overlap]))  # Concatène en une ligne
                chunk = chunk[-overlap:]  # Garde le chevauchement

        if chunk:
            chunks.append(" ".join(chunk))  # Ajoute le dernier chunk

        return chunks