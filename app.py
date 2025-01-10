import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Liste de mots-clés spécifiques à chaque domaine
sisr_keywords = ['réseau', 'serveur', 'sécurité', 'infrastructure', 'virtualisation']
slam_keywords = ['logiciel', 'application', 'développement', 'gestion de projet', 'site web']

# Poids des mots-clés en fonction de leur pertinence
sisr_weights = {word: 2 for word in sisr_keywords}
slam_weights = {word: 2 for word in slam_keywords}

# Prétraitement du texte : suppression des accents et des caractères spéciaux
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[àâä]', 'a', text)
    text = re.sub(r'[îï]', 'i', text)
    text = re.sub(r'[ôö]', 'o', text)
    text = re.sub(r'[ùûü]', 'u', text)
    return text

# Fonction pour calculer les scores pour chaque catégorie
def calculate_scores(description):
    description = preprocess_text(description)
    words = description.split()
    sisr_score = sum([sisr_weights.get(word, 1) for word in words])
    slam_score = sum([slam_weights.get(word, 1) for word in words])
    return sisr_score, slam_score

# Exemple d'utilisation du classifieur
description = "Développement d'une application de gestion de projet"
sisr_score, slam_score = calculate_scores(description)

if sisr_score > slam_score:
    print("Orientation : SISR")
else:
    print("Orientation : SLAM")
