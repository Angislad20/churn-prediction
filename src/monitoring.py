import json

def detecter_drift(df_processed, seuil_alerte=0.15):
    """
    Compare les statistiques d'un nouveau batch à la baseline d'entraînement.
    Retourne la liste des alertes déclenchées (écart relatif > seuil_alerte).
    """
    with open("models/baseline_stats.json") as f:
        baseline = json.load(f)

    alertes = []

    age_mean_nouveau = df_processed["Age"].mean()
    ecart_age = abs(age_mean_nouveau - baseline["Age_mean"]) / baseline["Age_mean"]
    if ecart_age > seuil_alerte:
        alertes.append(f"Age moyen : {age_mean_nouveau:.1f} vs {baseline['Age_mean']:.1f} attendu (écart {ecart_age:.1%})")

    balance_mean_nouveau = df_processed["Balance"].mean()
    ecart_balance = abs(balance_mean_nouveau - baseline["Balance_mean"]) / baseline["Balance_mean"]
    if ecart_balance > seuil_alerte:
        alertes.append(f"Balance moyenne : {balance_mean_nouveau:.0f} vs {baseline['Balance_mean']:.0f} attendu (écart {ecart_balance:.1%})")

    taux_germany_nouveau = df_processed["Geography_Germany"].mean()
    ecart_geo = abs(taux_germany_nouveau - baseline["taux_geo_germany"])
    if ecart_geo > seuil_alerte:
        alertes.append(f"Proportion Allemagne : {taux_germany_nouveau:.1%} vs {baseline['taux_geo_germany']:.1%} attendu")

    return alertes