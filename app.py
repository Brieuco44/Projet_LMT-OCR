from flask import Flask, jsonify
from routes.recognition_routes import recognition_bp
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hello word !"})

app.register_blueprint(recognition_bp, url_prefix='/recognition')

if __name__ == '__main__':
    app.run(debug=True)
