import pandas as pd
from src.preprocessing import regrouper_produits

def test_regrouper_produits_valeurs_basses():
    assert regrouper_produits(1) == "1"
    assert regrouper_produits(2) == "2"

def test_regrouper_produits_valeurs_hautes():
    assert regrouper_produits(3) == "3+"
    assert regrouper_produits(4) == "3+"