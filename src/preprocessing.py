import pandas as pd
import joblib

def charger_objets_preprocessing():
    """Charge les encodeurs entraînés depuis models/."""
    encoder_geo = joblib.load("models/encoder_geography.pkl")
    encoder_prod = joblib.load("models/encoder_products.pkl")
    return encoder_geo, encoder_prod


def regrouper_produits(x):
    """Reproduit le regroupement de NumOfProducts en 1/2/3+."""
    return "3+" if x >= 3 else str(x)


def preprocess_nouveaux_clients(df, encoder_geo, encoder_prod):
    """
    Applique le même preprocessing que celui utilisé à l'entraînement,
    sur un DataFrame de nouveaux clients à scorer.
    """
    df = df.copy()

    # Exclusions actées en Phase 1/2
    colonnes_a_exclure = ["RowNumber", "CustomerId", "Surname", "Gender"]
    df = df.drop(columns=[c for c in colonnes_a_exclure if c in df.columns])

    # Encodage Geography
    geo_encoded = encoder_geo.transform(df[["Geography"]])
    geo_cols = encoder_geo.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols, index=df.index)
    df = pd.concat([df.drop(columns=["Geography"]), geo_df], axis=1)

    # Regroupement + encodage NumOfProducts
    df["NumOfProducts_grp"] = df["NumOfProducts"].apply(regrouper_produits)
    df = df.drop(columns=["NumOfProducts"])
    prod_encoded = encoder_prod.transform(df[["NumOfProducts_grp"]])
    prod_cols = encoder_prod.get_feature_names_out(["NumOfProducts_grp"])
    prod_df = pd.DataFrame(prod_encoded, columns=prod_cols, index=df.index)
    df = pd.concat([df.drop(columns=["NumOfProducts_grp"]), prod_df], axis=1)

    return df