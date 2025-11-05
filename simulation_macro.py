"""
Simulations macroéconomiques et visualisations avancées
Analyse de l'impact des politiques fiscales sur la population
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from calculateur_impot import CalculateurImpot
from modele_populationnel import ModelePopulationnel

class SimulationMacro:
    """Classe pour les simulations macroéconomiques et visualisations"""
    
    def __init__(self):
        self.calc = CalculateurImpot()
        self.modele = ModelePopulationnel()
        
        # Configuration des graphiques
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def analyser_impact_reforme(self, scenarios: List[Dict], 
                               duree: int = 30) -> Dict:
        """
        Analyse l'impact de différents scénarios de réforme
        
        Args:
            scenarios: Liste des scénarios à analyser
            duree: Durée de la simulation en années
            
        Returns:
            Résultats détaillés pour chaque scénario
        """
        resultats = {}
        
        for scenario in scenarios:
            nom = scenario['nom']
            print(f"Analyse du scénario: {nom}")
            
            # Paramètres du scénario
            delta_tau = scenario.get('delta_tau', 0)
            rho = scenario.get('rho', 0)
            mode = scenario.get('mode', 'edo')
            
            # Simulation
            if mode == 'edo':
                resultat = self.modele.simuler_edo(T=duree, delta_tau=delta_tau, rho=rho)
            else:
                resultat = self.modele.simuler_markov(T=duree, delta_tau=delta_tau, rho=rho)
            
            # Calcul d'indicateurs supplémentaires
            resultat = self._calculer_indicateurs_avances(resultat)
            
            resultats[nom] = resultat
        
        return resultats
    
    def _calculer_indicateurs_avances(self, resultat: Dict) -> Dict:
        """
        Calcule des indicateurs avancés pour l'analyse
        
        Args:
            resultat: Résultat de simulation de base
            
        Returns:
            Résultat enrichi avec les indicateurs
        """
        try:
            # Calcul des taux de mobilité
            taux_mobilite = self._calculer_taux_mobilite(resultat)
        except Exception as e:
            print(f"Erreur calcul mobilité: {e}")
            taux_mobilite = 0.0
        
        try:
            # Calcul de l'effet redistributif
            effet_redistributif = self._calculer_effet_redistributif(resultat)
        except Exception as e:
            print(f"Erreur calcul effet redistributif: {e}")
            effet_redistributif = 0.0
        
        try:
            # Calcul de l'élasticité des recettes
            elasticite_recettes = self._calculer_elasticite_recettes(resultat)
        except Exception as e:
            print(f"Erreur calcul élasticité: {e}")
            elasticite_recettes = 0.0
        
        try:
            # Calcul du taux de pauvreté (tranche 1)
            taux_pauvrete = self._calculer_taux_pauvrete(resultat)
        except Exception as e:
            print(f"Erreur calcul taux pauvreté: {e}")
            taux_pauvrete = [0.0] * len(resultat.get('temps', [0]))
        
        resultat['taux_mobilite'] = taux_mobilite
        resultat['effet_redistributif'] = effet_redistributif
        resultat['elasticite_recettes'] = elasticite_recettes
        resultat['taux_pauvrete'] = taux_pauvrete
        
        return resultat
    
    def _calculer_taux_mobilite(self, resultat: Dict) -> float:
        """Calcule le taux de mobilité inter-tranches"""
        N = resultat['population_par_tranche']
        
        # Vérifier les dimensions du tableau
        if len(N.shape) == 2:
            # Cas EDO : N est (temps, tranches)
            N_total = np.sum(N, axis=1)
            
            # Calcul de la variance de la répartition (proxy de mobilité)
            proportions = N / N_total[:, np.newaxis]
            variance_proportions = np.var(proportions, axis=1)
            
            # Taux de mobilité = évolution de la variance
            if len(variance_proportions) > 1:
                mobilite = np.abs(variance_proportions[-1] - variance_proportions[0])
            else:
                mobilite = 0
        else:
            # Cas Markov : N est une liste de tableaux
            proportions_list = []
            for N_t in N:
                N_total_t = np.sum(N_t)
                if N_total_t > 0:
                    proportions_list.append(N_t / N_total_t)
                else:
                    proportions_list.append(np.zeros_like(N_t))
            
            if len(proportions_list) > 1:
                # Calcul de la variance entre le début et la fin
                var_initial = np.var(proportions_list[0])
                var_final = np.var(proportions_list[-1])
                mobilite = abs(var_final - var_initial)
            else:
                mobilite = 0
            
        return mobilite
    
    def _calculer_effet_redistributif(self, resultat: Dict) -> float:
        """Calcule l'effet redistributif de la politique"""
        gini_initial = resultat['gini'][0]
        gini_final = resultat['gini'][-1]
        
        # Effet redistributif = réduction des inégalités
        effet = gini_initial - gini_final
        
        return effet
    
    def _calculer_elasticite_recettes(self, resultat: Dict) -> float:
        """Calcule l'élasticité des recettes par rapport au taux"""
        recettes_initial = resultat['recettes_fiscales'][0]
        recettes_final = resultat['recettes_fiscales'][-1]
        
        if recettes_initial > 0:
            elasticite = (recettes_final - recettes_initial) / recettes_initial
        else:
            elasticite = 0
            
        return elasticite
    
    def _calculer_taux_pauvrete(self, resultat: Dict) -> List[float]:
        """Calcule le taux de pauvreté (proportion dans la première tranche)"""
        N = resultat['population_par_tranche']
        
        # Vérifier les dimensions du tableau
        if len(N.shape) == 2:
            # Cas EDO : N est (temps, tranches)
            N_total = np.sum(N, axis=1)
            taux_pauvrete = []
            for t in range(len(N_total)):
                if N_total[t] > 0:
                    taux = N[t, 0] / N_total[t] * 100  # Corrigé: N[t, 0] au lieu de N[0, t]
                    taux_pauvrete.append(taux)
                else:
                    taux_pauvrete.append(0)
        else:
            # Cas Markov : N est une liste de tableaux
            taux_pauvrete = []
            for N_t in N:
                N_total_t = np.sum(N_t)
                if N_total_t > 0:
                    taux = N_t[0] / N_total_t * 100
                    taux_pauvrete.append(taux)
                else:
                    taux_pauvrete.append(0)
                
        return taux_pauvrete
    
    def visualiser_comparaison_scenarios(self, resultats: Dict):
        """
        Visualise la comparaison entre différents scénarios
        
        Args:
            resultats: Dictionnaire avec les résultats de chaque scénario
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recettes fiscales', 'Indice de Gini', 
                          'Taux de pauvreté', 'Taux de mobilité'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (nom, resultat) in enumerate(resultats.items()):
            color = colors[i % len(colors)]
            
            # Recettes fiscales
            fig.add_trace(
                go.Scatter(x=resultat['temps'], y=resultat['recettes_fiscales'],
                          mode='lines', name=f'{nom} - Recettes',
                          line=dict(color=color, width=2)),
                row=1, col=1
            )
            
            # Indice de Gini
            fig.add_trace(
                go.Scatter(x=resultat['temps'], y=resultat['gini'],
                          mode='lines', name=f'{nom} - Gini',
                          line=dict(color=color, width=2)),
                row=1, col=2
            )
            
            # Taux de pauvreté
            fig.add_trace(
                go.Scatter(x=resultat['temps'], y=resultat['taux_pauvrete'],
                          mode='lines', name=f'{nom} - Pauvreté',
                          line=dict(color=color, width=2)),
                row=2, col=1
            )
            
            # Taux de mobilité (valeur finale)
            fig.add_trace(
                go.Scatter(x=[resultat['temps'][-1]], y=[resultat['taux_mobilite']],
                          mode='markers', name=f'{nom} - Mobilité',
                          marker=dict(color=color, size=10)),
                row=2, col=2
            )
        
        # Configuration des axes
        fig.update_xaxes(title_text="Temps (années)", row=2, col=1)
        fig.update_xaxes(title_text="Temps (années)", row=2, col=2)
        
        fig.update_yaxes(title_text="Recettes (€)", row=1, col=1)
        fig.update_yaxes(title_text="Indice de Gini", row=1, col=2)
        fig.update_yaxes(title_text="Taux de pauvreté (%)", row=2, col=1)
        fig.update_yaxes(title_text="Taux de mobilité", row=2, col=2)
        
        fig.update_layout(
            title_text="Comparaison des scénarios de politique fiscale",
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def creer_tableau_synthese(self, resultats: Dict) -> pd.DataFrame:
        """
        Crée un tableau de synthèse des résultats
        
        Args:
            resultats: Résultats de simulation
            
        Returns:
            DataFrame avec les indicateurs clés
        """
        donnees = []
        
        for nom, resultat in resultats.items():
            # Indicateurs finaux
            recettes_finales = resultat['recettes_fiscales'][-1]
            gini_final = resultat['gini'][-1]
            taux_pauvrete_final = resultat['taux_pauvrete'][-1]
            
            # Évolutions
            recettes_evolution = (recettes_finales - resultat['recettes_fiscales'][0]) / resultat['recettes_fiscales'][0] * 100
            gini_evolution = resultat['gini'][-1] - resultat['gini'][0]
            
            # Paramètres de la politique
            delta_tau = resultat.get('delta_tau', 0)
            rho = resultat.get('rho', 0)
            
            donnees.append({
                'Scénario': nom,
                'Δτ (hausse taux sup.)': f"{delta_tau*100:.1f}%",
                'ρ (redistribution)': f"{rho*100:.1f}%",
                'Recettes finales (M€)': f"{recettes_finales/1e6:.1f}",
                'Évolution recettes (%)': f"{recettes_evolution:+.1f}%",
                'Gini final': f"{gini_final:.3f}",
                'Δ Gini': f"{gini_evolution:+.3f}",
                'Taux pauvreté (%)': f"{taux_pauvrete_final:.1f}%",
                'Mobilité': f"{resultat['taux_mobilite']:.3f}",
                'Effet redistributif': f"{resultat['effet_redistributif']:+.3f}"
            })
        
        return pd.DataFrame(donnees)
    
    def analyser_effet_courbe_laffer(self, taux_hausses: List[float], 
                                   duree: int = 20) -> Dict:
        """
        Analyse l'effet de la courbe de Laffer (relation taux-revenus)
        
        Args:
            taux_hausses: Liste des hausses de taux à tester
            duree: Durée de simulation
            
        Returns:
            Résultats de l'analyse
        """
        resultats_laffer = {}
        
        for delta_tau in taux_hausses:
            resultat = self.modele.simuler_edo(T=duree, delta_tau=delta_tau, rho=0)
            
            recettes_initial = resultat['recettes_fiscales'][0]
            recettes_final = resultat['recettes_fiscales'][-1]
            
            resultats_laffer[delta_tau] = {
                'recettes_initiales': recettes_initial,
                'recettes_finales': recettes_final,
                'evolution_recettes': (recettes_final - recettes_initial) / recettes_initial * 100,
                'gini_evolution': resultat['gini'][-1] - resultat['gini'][0]
            }
        
        return resultats_laffer
    
    def visualiser_courbe_laffer(self, resultats_laffer: Dict):
        """
        Visualise la courbe de Laffer
        
        Args:
            resultats_laffer: Résultats de l'analyse Laffer
        """
        delta_taux = list(resultats_laffer.keys())
        evolutions_recettes = [r['evolution_recettes'] for r in resultats_laffer.values()]
        evolutions_gini = [r['gini_evolution'] for r in resultats_laffer.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Courbe de Laffer
        ax1.plot(delta_taux, evolutions_recettes, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Hausse du taux marginal supérieur')
        ax1.set_ylabel('Évolution des recettes (%)')
        ax1.set_title('Courbe de Laffer - Effet sur les recettes')
        ax1.grid(True, alpha=0.3)
        
        # Effet sur les inégalités
        ax2.plot(delta_taux, evolutions_gini, 'ro-', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Hausse du taux marginal supérieur')
        ax2.set_ylabel('Évolution de l\'indice de Gini')
        ax2.set_title('Effet sur les inégalités')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generer_rapport_executif(self, resultats: Dict) -> str:
        """
        Génère un rapport exécutif des simulations
        
        Args:
            resultats: Résultats de simulation
            
        Returns:
            Rapport sous forme de texte
        """
        tableau = self.creer_tableau_synthese(resultats)
        
        rapport = f"""
=== RAPPORT EXÉCUTIF - SIMULATION FISCALE ===

BARÈME UTILISÉ: 2024 (Source: service-public.gouv.fr)
PÉRIODE DE SIMULATION: {len(list(resultats.values())[0]['temps'])} années
POPULATION SIMULÉE: 1,000,000 habitants

=== RÉSULTATS PAR SCÉNARIO ===

{tableau.to_string(index=False)}

=== PRINCIPALES OBSERVATIONS ===

"""
        
        # Analyse automatique des résultats
        meilleur_recettes = tableau.loc[tableau['Évolution recettes (%)'].str.replace('%', '').str.replace('+', '').astype(float).idxmax()]
        meilleur_egalite = tableau.loc[tableau['Effet redistributif'].str.replace('+', '').astype(float).idxmax()]
        
        rapport += f"""
• Meilleur scénario pour les recettes: {meilleur_recettes['Scénario']} 
  (+{meilleur_recettes['Évolution recettes (%)']} recettes)
  
• Meilleur scénario pour l'égalité: {meilleur_egalite['Scénario']}
  (Δ Gini = {meilleur_egalite['Δ Gini']})

=== RECOMMANDATIONS ===

"""
        
        # Recommandations basées sur les résultats
        if float(meilleur_recettes['Évolution recettes (%)'].replace('%', '').replace('+', '')) > 5:
            rapport += "• La hausse des taux peut générer des recettes supplémentaires significatives\n"
        
        if float(meilleur_egalite['Effet redistributif'].replace('+', '')) > 0.01:
            rapport += "• Les politiques redistributives réduisent efficacement les inégalités\n"
        
        rapport += "• Un équilibre entre recettes et équité nécessite une approche nuancée\n"
        
        return rapport


# Exemples d'utilisation
if __name__ == "__main__":
    print("=== SIMULATIONS MACROÉCONOMIQUES ===\n")
    
    sim = SimulationMacro()
    
    # Définition des scénarios
    scenarios = [
        {
            'nom': 'Status quo',
            'delta_tau': 0,
            'rho': 0,
            'mode': 'edo'
        },
        {
            'nom': 'Hausse taux supérieur (+3%)',
            'delta_tau': 0.03,
            'rho': 0,
            'mode': 'edo'
        },
        {
            'nom': 'Hausse taux supérieur (+5%)',
            'delta_tau': 0.05,
            'rho': 0,
            'mode': 'edo'
        },
        {
            'nom': 'Redistribution (10%)',
            'delta_tau': 0,
            'rho': 0.1,
            'mode': 'edo'
        },
        {
            'nom': 'Réforme complète',
            'delta_tau': 0.03,
            'rho': 0.05,
            'mode': 'edo'
        }
    ]
    
    # Analyse des scénarios
    print("Analyse des différents scénarios...")
    resultats = sim.analyser_impact_reforme(scenarios, duree=25)
    
    # Tableau de synthèse
    tableau = sim.creer_tableau_synthese(resultats)
    print("\n=== TABLEAU DE SYNTHÈSE ===")
    print(tableau.to_string(index=False))
    
    # Analyse de la courbe de Laffer
    print("\nAnalyse de la courbe de Laffer...")
    taux_hausses = [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    resultats_laffer = sim.analyser_effet_courbe_laffer(taux_hausses)
    sim.visualiser_courbe_laffer(resultats_laffer)
    
    # Rapport exécutif
    rapport = sim.generer_rapport_executif(resultats)
    print(rapport)
    
    # Sauvegarde du rapport
    with open('rapport_simulation.txt', 'w', encoding='utf-8') as f:
        f.write(rapport)
    
    print("Rapport sauvegardé dans 'rapport_simulation.txt'")
