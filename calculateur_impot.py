"""
Calculateur d'impôt sur le revenu français - Modèle individuel
Barème 2024 - Source: service-public.gouv.fr
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class CalculateurImpot:
    """Calculateur d'impôt progressif par tranches conforme au barème 2024"""
    
    def __init__(self, annee: int = 2024):
        self.annee = annee
        self.bareme_2024 = {
            'tranches': [0, 11497, 29315, 83823, 180294],
            'taux': [0, 0.11, 0.30, 0.41, 0.45]
        }
        
        # Paramètres pour la décote et le plafonnement
        self.decote_seuil_celibataire = 1726
        self.decote_seuil_couple = 2854
        self.plafond_quotient_familial = 1576
        
    def calculer_parts(self, nb_adultes: int, nb_enfants: int) -> float:
        """
        Calcule le nombre de parts du quotient familial
        
        Args:
            nb_adultes: Nombre d'adultes (1 ou 2)
            nb_enfants: Nombre d'enfants à charge
            
        Returns:
            Nombre de parts
        """
        if nb_adultes not in [1, 2]:
            raise ValueError("Le nombre d'adultes doit être 1 ou 2")
            
        # Parts de base
        if nb_adultes == 1:
            parts = 1.0
        else:  # nb_adultes == 2
            parts = 2.0
            
        # Parts supplémentaires pour les enfants
        parts += nb_enfants * 0.5
        
        return parts
    
    def calculer_impot_par_part(self, revenu_par_part: float) -> float:
        """
        Calcule l'impôt pour une part selon le barème progressif
        
        Args:
            revenu_par_part: Revenu imposable par part
            
        Returns:
            Impôt par part
        """
        if revenu_par_part <= 0:
            return 0
            
        impot = 0
        tranches = self.bareme_2024['tranches']
        taux = self.bareme_2024['taux']
        
        for i in range(len(tranches) - 1):
            if revenu_par_part > tranches[i]:
                montant_imposable = min(revenu_par_part, tranches[i + 1]) - tranches[i]
                impot += montant_imposable * taux[i]
                
        # Dernière tranche (au-delà de 180294€)
        if revenu_par_part > tranches[-1]:
            montant_imposable = revenu_par_part - tranches[-1]
            impot += montant_imposable * taux[-1]
            
        return impot
    
    def appliquer_decote(self, impot_brut: float, nb_adultes: int) -> float:
        """
        Applique la décote si applicable
        
        Args:
            impot_brut: Impôt avant décote
            nb_adultes: Nombre d'adultes
            
        Returns:
            Impôt après décote
        """
        if nb_adultes == 1:
            seuil = self.decote_seuil_celibataire
            decote = seuil - 0.75 * impot_brut
        else:  # nb_adultes == 2
            seuil = self.decote_seuil_couple
            decote = seuil - 0.75 * impot_brut
            
        if decote > 0:
            return max(0, impot_brut - decote)
        else:
            return impot_brut
    
    def appliquer_plafonnement_quotient(self, impot_avant_plafonnement: float, 
                                      nb_enfants: int) -> float:
        """
        Applique le plafonnement du quotient familial
        
        Args:
            impot_avant_plafonnement: Impôt avant plafonnement
            nb_enfants: Nombre d'enfants à charge
            
        Returns:
            Impôt après plafonnement
        """
        if nb_enfants == 0:
            return impot_avant_plafonnement
            
        # Calcul de l'impôt sans quotient familial (1 part par adulte)
        impot_sans_quotient = self.calculer_impot_par_part(
            self.revenu_total / (2 if self.nb_adultes == 2 else 1)
        )
        
        # Avantage maximum du quotient familial
        avantage_max = nb_enfants * self.plafond_quotient_familial
        
        # Application du plafonnement
        impot_avec_plafonnement = impot_sans_quotient - avantage_max
        
        return max(impot_avec_plafonnement, impot_avant_plafonnement)
    
    def calculer_impot(self, revenu: float, nb_adultes: int, nb_enfants: int) -> Dict:
        """
        Calcule l'impôt total pour un foyer fiscal
        
        Args:
            revenu: Revenu imposable total
            nb_adultes: Nombre d'adultes (1 ou 2)
            nb_enfants: Nombre d'enfants à charge
            
        Returns:
            Dictionnaire avec tous les détails du calcul
        """
        # Stockage pour les méthodes qui en ont besoin
        self.revenu_total = revenu
        self.nb_adultes = nb_adultes
        
        # 1. Calcul du nombre de parts
        parts = self.calculer_parts(nb_adultes, nb_enfants)
        
        # 2. Revenu par part
        revenu_par_part = revenu / parts
        
        # 3. Impôt par part
        impot_par_part = self.calculer_impot_par_part(revenu_par_part)
        
        # 4. Impôt total avant décote
        impot_brut = parts * impot_par_part
        
        # 5. Application de la décote
        impot_apres_decote = self.appliquer_decote(impot_brut, nb_adultes)
        
        # 6. Application du plafonnement du quotient familial
        impot_final = self.appliquer_plafonnement_quotient(impot_apres_decote, nb_enfants)
        
        # Calcul des taux
        taux_marginal = self.calculer_taux_marginal(revenu_par_part)
        taux_moyen = impot_final / revenu if revenu > 0 else 0
        
        # Détail par tranches
        detail_tranches = self.calculer_detail_tranches(revenu_par_part)
        
        return {
            'revenu_total': revenu,
            'nb_adultes': nb_adultes,
            'nb_enfants': nb_enfants,
            'parts': parts,
            'revenu_par_part': revenu_par_part,
            'impot_par_part': impot_par_part,
            'impot_brut': impot_brut,
            'impot_apres_decote': impot_apres_decote,
            'impot_final': impot_final,
            'taux_marginal': taux_marginal,
            'taux_moyen': taux_moyen,
            'detail_tranches': detail_tranches
        }
    
    def calculer_taux_marginal(self, revenu_par_part: float) -> float:
        """Calcule le taux marginal d'imposition"""
        tranches = self.bareme_2024['tranches']
        taux = self.bareme_2024['taux']
        
        for i in range(len(tranches) - 1):
            if tranches[i] < revenu_par_part <= tranches[i + 1]:
                return taux[i]
                
        # Au-delà de la dernière tranche
        if revenu_par_part > tranches[-1]:
            return taux[-1]
            
        return taux[0]  # Tranche 0%
    
    def calculer_detail_tranches(self, revenu_par_part: float) -> List[Dict]:
        """Calcule le détail de l'imposition par tranche"""
        detail = []
        tranches = self.bareme_2024['tranches']
        taux = self.bareme_2024['taux']
        
        for i in range(len(tranches) - 1):
            limite_basse = tranches[i]
            limite_haute = tranches[i + 1]
            
            if revenu_par_part > limite_basse:
                montant_imposable = min(revenu_par_part, limite_haute) - limite_basse
                impot_tranche = montant_imposable * taux[i]
                
                detail.append({
                    'tranche': f"{limite_basse:,.0f}€ - {limite_haute:,.0f}€",
                    'taux': f"{taux[i]*100:.0f}%",
                    'montant_imposable': montant_imposable,
                    'impot_tranche': impot_tranche
                })
        
        # Dernière tranche
        if revenu_par_part > tranches[-1]:
            montant_imposable = revenu_par_part - tranches[-1]
            impot_tranche = montant_imposable * taux[-1]
            
            detail.append({
                'tranche': f"Plus de {tranches[-1]:,.0f}€",
                'taux': f"{taux[-1]*100:.0f}%",
                'montant_imposable': montant_imposable,
                'impot_tranche': impot_tranche
            })
            
        return detail
    
    def visualiser_bareme(self):
        """Visualise le barème d'imposition"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique du barème
        tranches = self.bareme_2024['tranches']
        taux = self.bareme_2024['taux']
        
        # Créer des segments pour chaque tranche
        x_values = []
        y_values = []
        
        for i in range(len(tranches) - 1):
            x_values.extend([tranches[i], tranches[i+1]])
            y_values.extend([taux[i]*100, taux[i]*100])
        
        # Dernière tranche (au-delà de 180294€)
        x_values.extend([tranches[-1], tranches[-1] * 1.5])
        y_values.extend([taux[-1]*100, taux[-1]*100])
        
        ax1.plot(x_values, y_values, 'b-', linewidth=2)
        ax1.set_xlabel('Revenu par part (€)')
        ax1.set_ylabel('Taux marginal (%)')
        ax1.set_title('Barème d\'imposition 2024')
        ax1.grid(True, alpha=0.3)
        
        # Taux moyen en fonction du revenu
        revenus = np.linspace(0, 300000, 1000)
        taux_moyens = []
        
        for revenu in revenus:
            if revenu > 0:
                impot = self.calculer_impot_par_part(revenu)
                taux_moyen = impot / revenu * 100
                taux_moyens.append(taux_moyen)
            else:
                taux_moyens.append(0)
        
        ax2.plot(revenus, taux_moyens, 'r-', linewidth=2)
        ax2.set_xlabel('Revenu par part (€)')
        ax2.set_ylabel('Taux moyen (%)')
        ax2.set_title('Taux moyen d\'imposition 2024')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def comparer_foyers(self, foyers: List[Dict]):
        """
        Compare l'imposition de différents foyers
        
        Args:
            foyers: Liste de dictionnaires avec 'revenu', 'nb_adultes', 'nb_enfants', 'nom'
        """
        resultats = []
        
        for foyer in foyers:
            resultat = self.calculer_impot(
                foyer['revenu'], 
                foyer['nb_adultes'], 
                foyer['nb_enfants']
            )
            resultat['nom'] = foyer['nom']
            resultats.append(resultat)
        
        # Création du DataFrame pour l'affichage
        df = pd.DataFrame([{
            'Foyer': r['nom'],
            'Revenu': f"{r['revenu_total']:,.0f}€",
            'Parts': r['parts'],
            'Revenu/part': f"{r['revenu_par_part']:,.0f}€",
            'Impôt': f"{r['impot_final']:,.0f}€",
            'Taux moyen': f"{r['taux_moyen']*100:.1f}%",
            'Taux marginal': f"{r['taux_marginal']*100:.0f}%"
        } for r in resultats])
        
        print(df.to_string(index=False))
        return resultats


# Tests et exemples d'utilisation
if __name__ == "__main__":
    calc = CalculateurImpot()
    
    # Test avec différents foyers
    foyers_test = [
        {'nom': 'Célibataire', 'revenu': 30000, 'nb_adultes': 1, 'nb_enfants': 0},
        {'nom': 'Couple sans enfants', 'revenu': 60000, 'nb_adultes': 2, 'nb_enfants': 0},
        {'nom': 'Couple 2 enfants', 'revenu': 80000, 'nb_adultes': 2, 'nb_enfants': 2},
        {'nom': 'Famille monoparentale 2 enfants', 'revenu': 45000, 'nb_adultes': 1, 'nb_enfants': 2}
    ]
    
    print("=== CALCULATEUR D'IMPÔT SUR LE REVENU 2024 ===\n")
    print("Comparaison de différents foyers fiscaux:")
    resultats = calc.comparer_foyers(foyers_test)
    
    print("\n=== DÉTAIL DU CALCUL POUR UN COUPLE AVEC 2 ENFANTS ===")
    detail = calc.calculer_impot(80000, 2, 2)
    print(f"Revenu total: {detail['revenu_total']:,.0f}€")
    print(f"Nombre de parts: {detail['parts']}")
    print(f"Revenu par part: {detail['revenu_par_part']:,.0f}€")
    print(f"Impôt final: {detail['impot_final']:,.0f}€")
    print(f"Taux moyen: {detail['taux_moyen']*100:.2f}%")
    print(f"Taux marginal: {detail['taux_marginal']*100:.0f}%")
    
    print("\nDétail par tranches:")
    for tranche in detail['detail_tranches']:
        print(f"  {tranche['tranche']} ({tranche['taux']}): "
              f"{tranche['montant_imposable']:,.0f}€ → {tranche['impot_tranche']:,.0f}€")
    
    # Visualisation du barème
    calc.visualiser_bareme()
