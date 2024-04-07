# Phishing Detector App

## Overview
This Flask web application detects phishing emails using a machine learning model. It analyzes email text to classify it as either 'safe' or 'phishing'. Built with Python, Flask, and Scikit-learn, it's a practical tool for anyone looking to enhance email security.

## Features
- **Phishing Email Detection**: Quickly classify emails as safe or phishing.
- **User-Friendly Interface**: Simple web interface for inputting and analyzing email text.
- **Machine Learning**: Employs a Naive Bayes classifier trained on a diverse dataset of emails.

## Dataset
The email dataset used for training the machine learning model is sourced from [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails?resource=download).

## Getting Started
To get a local copy up and running follow these simple steps.

### Prerequisites
- Python 3.6+
- pandas
- scikit-learn
- flask

### Installation
1. Clone the repo
   ```
   git clone https://github.com/mlmurphythree/Phishing-Detector-App.git
   ```
2. Navigate to the project directory

3. Run the app
   ```
   python app.py
   ```
Access the application through `http://127.0.0.1:5000` in your web browser.

