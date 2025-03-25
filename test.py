import time
from services.recognition_servicev2 import Recognition_service

# Exemple d'utilisation
if __name__ == "__main__":
    folder = "data/ca_signe_csp/"
    #"2_CA_Signé_CSP.pdf", "20_CA_Signé_CSP.pdf", "21_CA_Signé_CSP.pdf",
    pdf_paths = [ "2_CA_Signé_CSP.pdf", "20_CA_Signé_CSP.pdf", "21_CA_Signé_CSP.pdf","22_CA_Signé_CSP.pdf"]
    model_path = "./roberta_large_squad2_download"  # Modèle plus performant

    extractor = Recognition_service(model_path)

    listsquestions = [
        "Quel est le numéro d'identifiant ?",
        "Quel est le type de document : France Travail ou LIR25/LIN27 ?",
        "Quel est le nom et le prénom du référent ?",
        "Quel est le nom de l'Organisme ?",
        "Quelle est la date de l'adhésion CSP ?",
        "Quand l'accompagnement à t-il demarré ?",
        "Quand est la date de fin de l'accompagnement ?",
    ]

    for pdf_path in pdf_paths:
        pdf_path = folder + pdf_path
        extracted_text = extractor.extract_text_from_pdf(pdf_path)
        print("\nFichier de texte :", pdf_path)

        for question in listsquestions:
            try:
                answer = extractor.get_answer_from_text(extracted_text, question)
                if not answer.strip():  # Vérifie si la réponse est vide
                    answer = "Réponse non trouvée"
            except Exception as e:
                answer = f"Erreur : {str(e)}"  # Capture les erreurs sans interrompre le script

            print(f"{question} : {answer}")
            time.sleep(0.5)  # Petite pause pour éviter de sauter des lignes
