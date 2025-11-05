# Projet Impôt sur le Revenu - Modélisation et Simulation

Ce projet modélise et simule le calcul de l'impôt sur le revenu français à deux niveaux :

## Structure du projet

- `calculateur_impot.py` - Calculateur d'impôt individuel
- `modele_populationnel.py` - Modèle populationnel (EDO et Markov)
- `simulation_macro.py` - Simulations et visualisations
- `app_shiny.py` - Application Streamlit interactive
- `tests.py` - Tests unitaires
- `notebook_principal.ipynb` - Notebook principal
- `rapport_synthese.pdf` - Rapport de synthèse

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Calculateur individuel
```python
from calculateur_impot import CalculateurImpot

calc = CalculateurImpot()
resultat = calc.calculer_impot(revenu=50000, nb_adultes=2, nb_enfants=1)
```

### Application interactive
```bash
# Méthode recommandée (fonctionne même si Streamlit n'est pas dans PATH)
python -m streamlit run app_shiny.py

# OU utilisez le fichier batch
LANCER_APPLICATION.bat
```

## Barème utilisé
Barème 2024 - Source: service-public.gouv.fr
