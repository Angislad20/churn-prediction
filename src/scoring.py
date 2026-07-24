import pandas as pd
import joblib
from src.preprocessing import charger_objets_preprocessing, preprocess_nouveaux_clients


def scorer_nouveaux_clients(chemin_csv, seuil=0.3):
    """
    Prend un fichier CSV de nouveaux clients, applique le preprocessing
    et le modèle entraîné, retourne un DataFrame avec le score de risque
    ainsi que les données transformées (réutilisées pour la détection de drift).
    """
    df_raw = pd.read_csv(chemin_csv)
    customer_ids = df_raw["CustomerId"]

    encoder_geo, encoder_prod = charger_objets_preprocessing()
    model = joblib.load("models/random_forest.pkl")

    df_processed = preprocess_nouveaux_clients(df_raw, encoder_geo, encoder_prod)
    df_processed = df_processed.reindex(columns=model.feature_names_in_, fill_value=0)

    probabilites = model.predict_proba(df_processed)[:, 1]
    a_risque = (probabilites >= seuil).astype(int)

    resultats = pd.DataFrame({
        "CustomerId": customer_ids,
        "probabilite_churn": probabilites,
        "a_risque": a_risque
    })

    return resultats.sort_values("probabilite_churn", ascending=False), df_processed