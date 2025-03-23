from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from database import DB_URL, initBase

# Charger les variables d'environnement
load_dotenv()

# Initialiser l'application Flask
app = Flask(__name__)

# Activer CORS
CORS(app, resources={
    r"/*": {"origins": "*", "methods": ["POST", "PUT", "PATCH", "GET", "DELETE", "OPTIONS"], "allow_headers": "*"}})

app.secret_key = 'lemeilleurcoffredemotdepassedumondevoirmemedelunivers'

# Utiliser l'URL de connexion de database.py
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialiser la BDD
db = initBase(app)


# Route pour tester la connexion
@app.route('/')
def home():
    return jsonify({"message": "Hello world !"})


from routes.recognition_routes import recognition_bp
app.register_blueprint(recognition_bp, url_prefix='/recognition')

if __name__ == '__main__':
    app.run(debug=True)
