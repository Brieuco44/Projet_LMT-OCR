from flask import Flask, jsonify,request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from database import DB_URL, initBase, CLEF_SECRET
import services.recognition_service
# Charger les variables d'environnement
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


@app.route('/', methods=['POST'])
def analyseLivrable():
    if 'pdffile' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400

    pdffile = request.files['pdffile']
    typelivrable = request.form['typelivrable']

    rgntn_serv = services.recognition_service.Recognition_service(pdffile, typelivrable, db, typelivrable)

    return rgntn_serv.process(True)

if __name__ == '__main__':
    app.run(debug=True)
