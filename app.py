from flask import Flask,request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from database import DB_URL, initBase, CLEF_SECRET
import services.recognition_service
# Charger les variables d'environnement
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "deepset/roberta-large-squad2"
save_directory = "./roberta_large_squad2_download"

# Download and save only if the directory doesn't already exist
if not os.path.exists(save_directory):
    print("Downloading and saving model...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
else:
    print("Model already exists locally. Skipping download.")

load_dotenv()

# Initialiser l'application Flask
app = Flask(__name__)

# Activer CORS
CORS(app, resources={
    r"/*": {"origins": "*", "methods": ["POST", "PUT", "PATCH", "GET", "DELETE", "OPTIONS"], "allow_headers": "*"}})

app.secret_key = CLEF_SECRET

# Utiliser l'URL de connexion de database.py
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialiser la BDD
db = initBase(app)


@app.route('/analyse', methods=['POST'])
def analyseLivrable():

    if 'pdffile' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400

    pdffile = request.files['pdffile']

    typelivrable = request.form['typelivrable']

    rgntn_serv = services.recognition_service.Recognition_service(
        pdffile,
        typelivrable,
        db,
        model_path="./roberta_large_squad2_download",
        signature_model="./signature_model.pt"
    )

    rgntn_serv.draw_boxes_on_pdf(output_pdf_path="output_with_boxes1.pdf")


    return rgntn_serv.process(True)

if __name__ == '__main__':
    app.run(debug=True)
