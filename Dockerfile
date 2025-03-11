# Utilisation de l'image officielle Python
FROM python:3.11

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt .

# Installation des dépendances
RUN pip install -r requirements.txt

# Copie du code source
COPY . .

# Exposition du port Flask
EXPOSE 5000

# Commande par défaut
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
