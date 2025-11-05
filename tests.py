"""
Tests unitaires pour le calculateur d'impôt et le modèle populationnel
"""

import unittest
import numpy as np
from calculateur_impot import CalculateurImpot

class TestCalculateurImpot(unittest.TestCase):
    """Tests pour le calculateur d'impôt individuel"""
    
    def setUp(self):
        self.calc = CalculateurImpot()
    
    def test_calcul_parts(self):
        """Test du calcul du nombre de parts"""
        # Célibataire sans enfants
        self.assertEqual(self.calc.calculer_parts(1, 0), 1.0)
        
        # Couple sans enfants
        self.assertEqual(self.calc.calculer_parts(2, 0), 2.0)
        
        # Couple avec 2 enfants
        self.assertEqual(self.calc.calculer_parts(2, 2), 3.0)
        
        # Famille monoparentale avec 2 enfants
        self.assertEqual(self.calc.calculer_parts(1, 2), 2.0)
    
    def test_impot_nul_revenu_zero(self):
        """Test que l'impôt est nul pour un revenu de 0"""
        resultat = self.calc.calculer_impot(0, 1, 0)
        self.assertEqual(resultat['impot_final'], 0)
        self.assertEqual(resultat['taux_moyen'], 0)
    
    def test_continuite_aux_seuils(self):
        """Test de la continuité du calcul aux seuils des tranches"""
        # Test autour du premier seuil (11497€)
        resultat1 = self.calc.calculer_impot(11497, 1, 0)
        resultat2 = self.calc.calculer_impot(11498, 1, 0)
        
        # La différence d'impôt doit être minime (juste 1€ à 11%)
        diff_impot = abs(resultat2['impot_final'] - resultat1['impot_final'])
        self.assertLess(diff_impot, 1.0)
    
    def test_monotonie_taux(self):
        """Test que le taux moyen est croissant avec le revenu"""
        revenus = [10000, 20000, 40000, 80000, 150000]
        taux_moyens = []
        
        for revenu in revenus:
            resultat = self.calc.calculer_impot(revenu, 1, 0)
            taux_moyens.append(resultat['taux_moyen'])
        
        # Vérification que les taux sont croissants
        for i in range(1, len(taux_moyens)):
            self.assertLessEqual(taux_moyens[i-1], taux_moyens[i])
    
    def test_coherence_bareme(self):
        """Test de cohérence du barème"""
        # Revenu dans la première tranche (0%)
        resultat = self.calc.calculer_impot(10000, 1, 0)
        self.assertEqual(resultat['taux_marginal'], 0)
        
        # Revenu dans la deuxième tranche (11%)
        resultat = self.calc.calculer_impot(20000, 1, 0)
        self.assertEqual(resultat['taux_marginal'], 0.11)
        
        # Revenu dans la troisième tranche (30%)
        resultat = self.calc.calculer_impot(50000, 1, 0)
        self.assertEqual(resultat['taux_marginal'], 0.30)
    
    def test_effet_quotient_familial(self):
        """Test de l'effet du quotient familial"""
        # Célibataire avec 40k€
        resultat1 = self.calc.calculer_impot(40000, 1, 0)
        
        # Couple avec 80k€ (même revenu par part)
        resultat2 = self.calc.calculer_impot(80000, 2, 0)
        
        # L'impôt par part doit être identique
        self.assertAlmostEqual(resultat1['impot_par_part'], resultat2['impot_par_part'], places=2)
    
    def test_decote(self):
        """Test de l'application de la décote"""
        # Revenu faible pour déclencher la décote
        resultat = self.calc.calculer_impot(25000, 1, 0)
        
        # L'impôt final doit être inférieur à l'impôt brut
        self.assertLess(resultat['impot_final'], resultat['impot_brut'])
    
    def test_cas_reels(self):
        """Test avec des cas réels typiques"""
        # Célibataire avec revenu moyen
        resultat = self.calc.calculer_impot(35000, 1, 0)
        self.assertGreater(resultat['impot_final'], 0)
        self.assertLess(resultat['taux_moyen'], 0.15)  # Taux moyen raisonnable
        
        # Couple avec enfants
        resultat = self.calc.calculer_impot(70000, 2, 2)
        self.assertGreater(resultat['parts'], 2)
        self.assertGreater(resultat['impot_final'], 0)


class TestModelePopulationnel(unittest.TestCase):
    """Tests pour le modèle populationnel (à implémenter)"""
    
    def test_conservation_population(self):
        """Test que la population totale est conservée"""
        # À implémenter avec le modèle populationnel
        pass
    
    def test_probabilites_markov(self):
        """Test que les probabilités de la chaîne de Markov somment à 1"""
        # À implémenter avec le modèle populationnel
        pass
    
    def test_coherence_transitions(self):
        """Test de cohérence des transitions entre tranches"""
        # À implémenter avec le modèle populationnel
        pass


def run_tests():
    """Lance tous les tests"""
    print("=== TESTS DU CALCULATEUR D'IMPÔT ===\n")
    
    # Tests unitaires
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Tests de cohérence supplémentaires
    print("\n=== TESTS DE COHÉRENCE SUPPLÉMENTAIRES ===")
    calc = CalculateurImpot()
    
    # Test avec des valeurs extrêmes
    print("Test avec revenu très élevé...")
    resultat = calc.calculer_impot(1000000, 1, 0)
    print(f"Revenu: 1M€, Impôt: {resultat['impot_final']:,.0f}€, "
          f"Taux moyen: {resultat['taux_moyen']*100:.1f}%")
    
    # Test de la décote
    print("\nTest de la décote...")
    resultat = calc.calculer_impot(20000, 1, 0)
    print(f"Revenu: 20k€, Impôt brut: {resultat['impot_brut']:,.0f}€, "
          f"Impôt final: {resultat['impot_final']:,.0f}€")
    
    if resultat['impot_final'] < resultat['impot_brut']:
        print("✓ Décote appliquée correctement")
    else:
        print("✗ Décote non appliquée")
    
    print("\n=== TOUS LES TESTS TERMINÉS ===")


if __name__ == "__main__":
    run_tests()
