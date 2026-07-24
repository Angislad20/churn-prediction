import pandas as pd
from datetime import datetime
from src.scoring import scorer_nouveaux_clients
from src.monitoring import detecter_drift

resultats, df_processed = scorer_nouveaux_clients("data/raw/nouveaux_clients.csv")
resultats.to_csv("outputs/clients_a_risque.csv", index=False)

alertes = detecter_drift(df_processed)

if alertes:
    print("Alertes de dérive détectées :")
    for a in alertes:
        print(f"  - {a}")
else:
    print("Aucune dérive significative détectée.")

ligne_historique = pd.DataFrame([{
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "nb_clients": len(resultats),
    "nb_a_risque": resultats["a_risque"].sum(),
    "taux_a_risque": resultats["a_risque"].mean(),
    "nb_alertes_drift": len(alertes)
}])

try:
    historique = pd.read_csv("outputs/historique_scoring.csv")
    historique = pd.concat([historique, ligne_historique], ignore_index=True)
except FileNotFoundError:
    historique = ligne_historique

historique.to_csv("outputs/historique_scoring.csv", index=False)

print(f"{resultats['a_risque'].sum()} clients identifiés à risque sur {len(resultats)}")