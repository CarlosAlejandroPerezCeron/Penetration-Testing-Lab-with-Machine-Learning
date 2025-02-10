import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    data = pd.read_csv('data/sample_network_data.csv')
    X = data.drop('anomaly', axis=1)
    y = data['anomaly']

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, 'models/trained_model.pkl')
    print("Modelo entrenado y guardado.")

if __name__ == "__main__":
    train_model()