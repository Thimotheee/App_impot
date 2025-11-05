"""
Modèle populationnel pour la simulation de l'impact des politiques fiscales
Deux approches : EDO (Équations Différentielles Ordinaires) et Chaîne de Markov
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from calculateur_impot import CalculateurImpot

class ModelePopulationnel:
    """Modèle pour simuler l'évolution d'une population entre les tranches d'imposition"""
    
    def __init__(self, k: int = 5):
        """
        Initialise le modèle populationnel
        
        Args:
            k: Nombre de tranches d'imposition
        """
        self.k = k  # Nombre de tranches
        self.calc = CalculateurImpot()
        
        # Paramètres du modèle
        self.g = 0.02  # Croissance économique annuelle
        self.pi = 0.02  # Inflation annuelle
        self.alpha = 0.1  # Propension à monter de tranche
        self.beta = 0.05  # Propension à descendre de tranche
        
        # Seuils des tranches (revenu par part)
        self.seuils = self.calc.bareme_2024['tranches']
        
        # Revenus moyens par tranche (initialisation)
        self.mu_init = [8000, 22000, 55000, 130000, 250000]
        
        # Matrice de transition pour Markov
        self.Q = self._initialiser_matrice_transition()
        
    def _initialiser_matrice_transition(self) -> np.ndarray:
        """Initialise la matrice de transition de la chaîne de Markov"""
        Q = np.zeros((self.k, self.k))
        
        # Probabilités de rester dans la même tranche
        for i in range(self.k):
            Q[i, i] = 0.8  # 80% restent dans la même tranche
        
        # Probabilités de transition vers les tranches adjacentes
        for i in range(self.k):
            if i > 0:  # Peut descendre
                Q[i, i-1] = self.beta
            if i < self.k - 1:  # Peut monter
                Q[i, i+1] = self.alpha
        
        # Normalisation pour que chaque ligne somme à 1
        for i in range(self.k):
            Q[i, :] = Q[i, :] / np.sum(Q[i, :])
        
        return Q
    
    def _initialiser_population(self, N_total: int = 1000000) -> np.ndarray:
        """
        Initialise la répartition de la population entre les tranches
        
        Args:
            N_total: Population totale
            
        Returns:
            Vecteur N avec la répartition initiale
        """
        # Répartition approximative basée sur les données INSEE
        # Plus de personnes dans les tranches basses
        proportions = [0.25, 0.35, 0.25, 0.12, 0.03]
        
        N = np.array([int(p * N_total) for p in proportions])
        N[-1] = N_total - np.sum(N[:-1])  # Ajustement pour la somme exacte
        
        return N
    
    def _calculer_taux_transition_edo(self, N: np.ndarray, t: float, 
                                    delta_tau: float = 0, rho: float = 0) -> np.ndarray:
        """
        Calcule les taux de transition pour le modèle EDO
        
        Args:
            N: Vecteur de population par tranche
            t: Temps
            delta_tau: Variation du taux marginal supérieur
            rho: Part redistribuée aux tranches basses
            
        Returns:
            Dérivée dN/dt
        """
        dN_dt = np.zeros(self.k)
        
        # Effet de la croissance économique
        for i in range(self.k):
            # Croissance du revenu moyen dans la tranche
            mu_i = self.mu_init[i] * np.exp(self.g * t)
            
            # Probabilité de monter (fonction du revenu et de la croissance)
            prob_monter = self.alpha * (1 + delta_tau * (i == self.k-1))
            
            # Probabilité de descendre (fonction de l'inflation et redistribution)
            prob_descendre = self.beta * (1 - rho * (i < 2))
            
            # Flux sortants
            dN_dt[i] -= N[i] * (prob_monter + prob_descendre)
            
            # Flux entrants depuis les tranches adjacentes
            if i > 0:
                dN_dt[i] += N[i-1] * prob_monter
            if i < self.k - 1:
                dN_dt[i] += N[i+1] * prob_descendre
        
        return dN_dt
    
    def simuler_edo(self, T: int = 50, delta_tau: float = 0.05, 
                   rho: float = 0.1, N_total: int = 1000000) -> Dict:
        """
        Simulation avec le modèle EDO
        
        Args:
            T: Période de simulation (années)
            delta_tau: Hausse du taux marginal supérieur
            rho: Part redistribuée aux tranches basses
            N_total: Population totale
            
        Returns:
            Dictionnaire avec les résultats de simulation
        """
        # Conditions initiales
        N0 = self._initialiser_population(N_total)
        
        # Temps de simulation
        t = np.linspace(0, T, T+1)
        
        # Intégration des EDO
        def dN_dt(N, t):
            return self._calculer_taux_transition_edo(N, t, delta_tau, rho)
        
        N_t = odeint(dN_dt, N0, t)
        
        # Calcul des métriques
        revenus_totaux = []
        recettes_fiscales = []
        gini_t = []
        
        for i in range(len(t)):
            # Revenus totaux par tranche
            revenus_par_tranche = []
            recettes_par_tranche = []
            
            for j in range(self.k):
                if N_t[i, j] > 0:
                    # Revenu moyen dans la tranche (avec croissance)
                    mu_j = self.mu_init[j] * np.exp(self.g * t[i])
                    revenu_total_j = N_t[i, j] * mu_j
                    revenus_par_tranche.append(revenu_total_j)
                    
                    # Calcul de l'impôt moyen dans cette tranche
                    impot_moyen = self.calc.calculer_impot_par_part(mu_j)
                    recettes_j = N_t[i, j] * impot_moyen
                    recettes_par_tranche.append(recettes_j)
            
            revenus_totaux.append(sum(revenus_par_tranche))
            recettes_fiscales.append(sum(recettes_par_tranche))
            
            # Calcul de l'indice de Gini (simplifié)
            gini_t.append(self._calculer_gini_simplifie(N_t[i], self.mu_init))
        
        return {
            'temps': t,
            'population_par_tranche': N_t,
            'revenus_totaux': revenus_totaux,
            'recettes_fiscales': recettes_fiscales,
            'gini': gini_t,
            'delta_tau': delta_tau,
            'rho': rho
        }
    
    def simuler_markov(self, T: int = 50, delta_tau: float = 0.05, 
                      rho: float = 0.1, N_total: int = 1000000) -> Dict:
        """
        Simulation avec la chaîne de Markov
        
        Args:
            T: Période de simulation (années)
            delta_tau: Hausse du taux marginal supérieur
            rho: Part redistribuée aux tranches basses
            N_total: Population totale
            
        Returns:
            Dictionnaire avec les résultats de simulation
        """
        # Conditions initiales
        N = self._initialiser_population(N_total)
        
        # Matrice de transition modifiée par la politique fiscale
        Q_modifiee = self._modifier_matrice_transition(delta_tau, rho)
        
        # Simulation temporelle
        N_t = [N.copy()]
        revenus_totaux = []
        recettes_fiscales = []
        gini_t = []
        
        for t in range(1, T + 1):
            # Évolution stochastique
            N_new = np.zeros(self.k)
            for i in range(self.k):
                for j in range(self.k):
                    N_new[j] += N[i] * Q_modifiee[i, j]
            
            N = N_new.astype(int)  # Population entière
            N_t.append(N.copy())
            
            # Calcul des métriques
            revenus_par_tranche = []
            recettes_par_tranche = []
            
            for j in range(self.k):
                if N[j] > 0:
                    # Revenu moyen dans la tranche (avec croissance)
                    mu_j = self.mu_init[j] * np.exp(self.g * t)
                    revenu_total_j = N[j] * mu_j
                    revenus_par_tranche.append(revenu_total_j)
                    
                    # Calcul de l'impôt moyen dans cette tranche
                    impot_moyen = self.calc.calculer_impot_par_part(mu_j)
                    recettes_j = N[j] * impot_moyen
                    recettes_par_tranche.append(recettes_j)
            
            revenus_totaux.append(sum(revenus_par_tranche))
            recettes_fiscales.append(sum(recettes_par_tranche))
            
            # Calcul de l'indice de Gini
            gini_t.append(self._calculer_gini_simplifie(N, self.mu_init))
        
        return {
            'temps': np.arange(T + 1),
            'population_par_tranche': np.array(N_t),
            'revenus_totaux': revenus_totaux,
            'recettes_fiscales': recettes_fiscales,
            'gini': gini_t,
            'delta_tau': delta_tau,
            'rho': rho,
            'matrice_transition': Q_modifiee
        }
    
    def _modifier_matrice_transition(self, delta_tau: float, rho: float) -> np.ndarray:
        """
        Modifie la matrice de transition selon la politique fiscale
        
        Args:
            delta_tau: Hausse du taux marginal supérieur
            rho: Part redistribuée aux tranches basses
            
        Returns:
            Matrice de transition modifiée
        """
        Q_modifiee = self.Q.copy()
        
        # Effet de la hausse du taux supérieur (réduction de la mobilité ascendante)
        if delta_tau > 0:
            # Réduction des transitions vers la tranche supérieure
            for i in range(self.k - 1):
                Q_modifiee[i, i+1] *= (1 - delta_tau)
                Q_modifiee[i, i] += Q_modifiee[i, i+1] * delta_tau  # Réajustement
        
        # Effet de la redistribution (augmentation de la mobilité ascendante pour les tranches basses)
        if rho > 0:
            # Augmentation des transitions ascendantes pour les 2 premières tranches
            for i in range(min(2, self.k - 1)):
                Q_modifiee[i, i+1] *= (1 + rho)
                Q_modifiee[i, i] -= Q_modifiee[i, i+1] * rho / (1 + rho)  # Réajustement
        
        # Normalisation
        for i in range(self.k):
            Q_modifiee[i, :] = Q_modifiee[i, :] / np.sum(Q_modifiee[i, :])
        
        return Q_modifiee
    
    def _calculer_gini_simplifie(self, N: np.ndarray, mu: List[float]) -> float:
        """
        Calcule un indice de Gini simplifié
        
        Args:
            N: Population par tranche
            mu: Revenus moyens par tranche
            
        Returns:
            Indice de Gini (0 = égalité parfaite, 1 = inégalité maximale)
        """
        # Calcul des revenus cumulés
        revenus = N * np.array(mu)
        total_revenu = np.sum(revenus)
        total_population = np.sum(N)
        
        if total_revenu == 0:
            return 0
        
        # Tri par revenu moyen
        indices_tries = np.argsort(mu)
        revenus_tries = revenus[indices_tries]
        population_tries = N[indices_tries]
        
        # Calcul de l'aire sous la courbe de Lorenz
        aire_lorenz = 0
        cumul_revenu = 0
        cumul_population = 0
        
        for i in range(len(revenus_tries)):
            if population_tries[i] > 0:
                cumul_revenu += revenus_tries[i]
                cumul_population += population_tries[i]
                
                # Aire du trapèze
                aire_lorenz += cumul_revenu * population_tries[i] / total_population
        
        # Gini = 1 - 2 * aire_lorenz / total_revenu
        gini = 1 - 2 * aire_lorenz / (total_revenu * total_population)
        
        return max(0, min(1, gini))  # Borné entre 0 et 1
    
    def comparer_politiques(self, politiques: List[Dict], mode: str = 'edo') -> Dict:
        """
        Compare différentes politiques fiscales
        
        Args:
            politiques: Liste de dictionnaires avec les paramètres de chaque politique
            mode: 'edo' ou 'markov'
            
        Returns:
            Dictionnaire avec les résultats de toutes les politiques
        """
        resultats = {}
        
        for i, politique in enumerate(politiques):
            nom = politique.get('nom', f'Politique_{i+1}')
            delta_tau = politique.get('delta_tau', 0)
            rho = politique.get('rho', 0)
            
            if mode == 'edo':
                resultat = self.simuler_edo(delta_tau=delta_tau, rho=rho)
            else:
                resultat = self.simuler_markov(delta_tau=delta_tau, rho=rho)
            
            resultats[nom] = resultat
        
        return resultats
    
    def visualiser_evolution(self, resultats: Dict, titre: str = "Évolution de la population"):
        """
        Visualise l'évolution de la population et des métriques
        
        Args:
            resultats: Résultats de simulation
            titre: Titre du graphique
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Évolution de la population par tranche
        ax1 = axes[0, 0]
        for i in range(self.k):
            ax1.plot(resultats['temps'], resultats['population_par_tranche'][:, i], 
                    label=f'Tranche {i+1}', linewidth=2)
        ax1.set_xlabel('Temps (années)')
        ax1.set_ylabel('Population')
        ax1.set_title('Évolution de la population par tranche')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Recettes fiscales
        ax2 = axes[0, 1]
        ax2.plot(resultats['temps'], resultats['recettes_fiscales'], 
                'r-', linewidth=2, label='Recettes fiscales')
        ax2.set_xlabel('Temps (années)')
        ax2.set_ylabel('Recettes fiscales (€)')
        ax2.set_title('Évolution des recettes fiscales')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Indice de Gini
        ax3 = axes[1, 0]
        ax3.plot(resultats['temps'], resultats['gini'], 
                'g-', linewidth=2, label='Indice de Gini')
        ax3.set_xlabel('Temps (années)')
        ax3.set_ylabel('Indice de Gini')
        ax3.set_title('Évolution des inégalités')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Répartition finale
        ax4 = axes[1, 1]
        population_finale = resultats['population_par_tranche'][-1]
        parts = [f'Tranche {i+1}' for i in range(self.k)]
        ax4.pie(population_finale, labels=parts, autopct='%1.1f%%')
        ax4.set_title('Répartition finale de la population')
        
        plt.suptitle(titre, fontsize=16)
        plt.tight_layout()
        plt.show()


# Exemples d'utilisation
if __name__ == "__main__":
    print("=== MODÈLE POPULATIONNEL ===\n")
    
    modele = ModelePopulationnel()
    
    # Simulation de base (sans réforme)
    print("Simulation de base (sans réforme fiscale)...")
    resultat_base = modele.simuler_edo(T=20)
    
    # Simulation avec hausse du taux supérieur
    print("Simulation avec hausse du taux supérieur (+5%)...")
    resultat_reforme = modele.simuler_edo(T=20, delta_tau=0.05)
    
    # Comparaison des politiques
    politiques = [
        {'nom': 'Status quo', 'delta_tau': 0, 'rho': 0},
        {'nom': 'Hausse taux supérieur', 'delta_tau': 0.05, 'rho': 0},
        {'nom': 'Redistribution', 'delta_tau': 0, 'rho': 0.1},
        {'nom': 'Réforme complète', 'delta_tau': 0.05, 'rho': 0.1}
    ]
    
    print("Comparaison de différentes politiques...")
    resultats_comparaison = modele.comparer_politiques(politiques)
    
    # Affichage des résultats finaux
    print("\n=== RÉSULTATS FINAUX (Année 20) ===")
    for nom, resultat in resultats_comparaison.items():
        recettes_finales = resultat['recettes_fiscales'][-1]
        gini_final = resultat['gini'][-1]
        print(f"{nom}: Recettes = {recettes_finales:,.0f}€, Gini = {gini_final:.3f}")
    
    # Visualisation
    modele.visualiser_evolution(resultat_reforme, "Impact d'une hausse du taux supérieur")
