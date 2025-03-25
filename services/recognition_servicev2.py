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

    def extract_text_from_pdf(self, pdf_path):
        """Extraction du texte à partir d'un PDF avec OCR optimisé."""
        images = convert_from_path(pdf_path, dpi=600)  # Ajuster selon besoin
        text = ''
        for img in images:
            text += pytesseract.image_to_string(img, lang="fra").replace("\n"," ")
        return text

    def get_answer_from_text(self, text, question, score_threshold=0.2):
        """
        Découpe le texte et sélectionne la meilleure réponse trouvée dans les chunks
        si le score dépasse le seuil, et vérifie que la réponse se trouve dans le texte.
        """

        best_answer = None
        best_score = 0.0

        results = self.question_answerer(question=question, context=text, top_k=3)
        best_local_result = max(results, key=lambda r: r['score'])

        print(best_local_result)

        # Garder la réponse la plus élevée sur l'ensemble des chunks
        if best_local_result['score'] > best_score:
            best_answer = best_local_result['answer']
            best_score = best_local_result['score']

        # Si le score est insuffisant, on rejette la réponse
        if best_score < score_threshold:
            return "Réponse non trouvée"

        return best_answer.strip() if best_answer else "Réponse non trouvée"

    @staticmethod
    def chunk_text(text, max_words=4000, overlap=200):  # Augmenter la taille des chunks
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

        print(chunks)
        return chunks