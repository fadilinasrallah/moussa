Here is the structured extraction of the text from the provided document:

## Informations Générales

* **Établissement :** ENSA Tanger, Filière Génie Cybersécurité et Cyberintelligence.


* **Titre du projet :** Système de détection automatique de données dissimulées dans les images et vidéos numériques par analyse de texture et classification automatique.


* **Année universitaire :** 2025-2026.


* **Équipe projet :** Zirar Chaimae, Fadili Nasrallah, Cheikh Mariam, Toukour Nouha, Oussafi Ilyas.


* **Encadrant académique :** M. Ahmed Moussa.



---

## Introduction et Problématique

* La stéganographie consiste à masquer la présence d'informations dans des supports numériques, ce qui peut être exploité en cybersécurité pour dissimuler des communications malveillantes.


* La problématique centrale consiste à déterminer comment exploiter les caractéristiques de texture d'une image pour détecter ces données cachées de manière fiable et automatisée.



---

## Objectifs du Projet

* L'objectif principal est de développer un système capable de détecter automatiquement les données cachées dans une image numérique.


* Il est nécessaire d'étudier les méthodes de stéganographie LSB et de mettre en œuvre des techniques d'analyse de texture.


* Le système doit extraire des caractéristiques discriminantes et implémenter des modèles de classification automatique.


* Le projet requiert l'évaluation des performances du système et la structuration de l'application selon la programmation orientée objet (POO).



---

## Périmètre du Projet

| Éléments inclus | Éléments exclus |
| --- | --- |
| Images aux formats standards (JPEG, PNG)

 | Techniques avancées (domaine fréquentiel)

 |
| Détection de stéganographie LSB

 | Attaques adversariales sophistiquées

 |
| Analyse par descripteurs de texture

 | Traitement en temps réel à grande échelle

 |
| Vidéos numériques (MP4, AVI)

 | Analyse en temps réel de flux vidéo

 |
| Extraction et analyse de frames vidéo

 | Stéganographie vidéo dans le domaine compressé

 |

---

## Approche Technique et Architecture

* **Prétraitement :** Les données d'entrée sont homogénéisées par une conversion en niveaux de gris, un redimensionnement et une normalisation.


* **Extraction de caractéristiques :** Le système s'appuie sur les Local Binary Patterns (LBP) pour les motifs locaux, la matrice GLCM pour les relations spatiales, et les histogrammes de niveaux de gris.


* **Traitement vidéo :** Les vidéos sont décomposées en frames individuelles grâce à la bibliothèque OpenCV.


* **Classification :** Le pipeline utilise les algorithmes Support Vector Machine (SVM) et Random Forest pour identifier les anomalies.



---

## Dépendances et Exigences

* **Bibliothèques :** Le développement repose sur OpenCV (traitement d'images/vidéos), NumPy (calcul numérique) et Scikit-learn (classification).


* **Exigences non fonctionnelles :** Le code doit être modulaire, lisible, reproductible, robuste face aux variations d'images, et exécutable dans un temps raisonnable.


* **Contraintes techniques :** Le projet se limite aux méthodes LSB (domaine spatial) et impose l'utilisation de bibliothèques open-source.



---

## Évaluation, Livrables et Perspectives

* **Métriques :** Les modèles seront évalués selon leur Exactitude (Accuracy), Précision (Precision), Rappel (Recall) et via une Matrice de confusion.


* **Livrables attendus :** Le rendu final inclura le code source, un rapport technique, le jeu de données, les résultats expérimentaux et un support de présentation.


* **Perspectives d'amélioration :** Le système pourra être étendu au domaine fréquentiel, intégrer des modèles de deep learning (CNN, autoencodeurs) ou se doter d'une interface graphique.