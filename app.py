from flask import Flask, jsonify
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
import os
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST", "PUT", "PATCH", "GET", "DELETE", "OPTIONS"], "allow_headers": "*"}})

app.secret_key = 'lemeilleurcoffredemotdepassedumondevoirmemedelunivers'

# Chargement des variables d'environnements
load_dotenv()

server = os.getenv('MYSQL_HOST')+":"+os.getenv('MYSQL_PORT')


param_bdd = "mysql+pymysql://"+os.getenv('MYSQL_USER')+":"+os.getenv('MYSQL_PASSWORD')+"@"+server+"/"+os.getenv('MYSQL_DATABASE')
app.config['SQLALCHEMY_DATABASE_URI'] = param_bdd

# désactiver car gourmand en ressources
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

@app.route('/')
def home():
    return jsonify({"message": "Hello word !"})

from routes.recognition_routes import recognition_bp

app.register_blueprint(recognition_bp, url_prefix='/recognition')

if __name__ == '__main__':
    app.run(debug=True)
