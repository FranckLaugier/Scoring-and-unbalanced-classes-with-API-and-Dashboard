import pandas as pd
import pickle
import joblib
from flask import Flask, jsonify

# Chargement des données
# Les données
X_test = pd.read_csv('X_test.csv')
# Le modèle
model = joblib.load('model_fitted_home_credit.joblib')
# Les features
features_70 = pickle.load(open('liste_70_feature_importances.pickle', 'rb'))

# Définition de l'application
app = Flask(__name__)

# Définition du chemin root
@app.route('/')
def index():
    return 'Bienvenue'

# Définition du chemin prédiction/id_client
@app.route("/prediction/<int:ID>", methods=["GET"])
def make_prediction(ID):
    df_client = X_test[X_test['SK_ID_CURR'] == ID]
    prediction = model.predict_proba(df_client[features_70])
    print("""Le score pour le client {} est de {} """.format(ID, round(float(prediction[:,1]), 2)))
    prediction = prediction[0].tolist()
    print('Prediction : ',  str(prediction))
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run( debug=True)
