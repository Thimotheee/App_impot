"""
Application Streamlit interactive pour le projet d'impôt sur le revenu
Interface utilisateur avec deux onglets : calculateur individuel et simulation populationnelle
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from calculateur_impot import CalculateurImpot
from modele_populationnel import ModelePopulationnel
from simulation_macro import SimulationMacro

# Configuration de la page
st.set_page_config(
    page_title="Simulateur d'Impôt sur le Revenu",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Fonction principale de l'application"""
    
    # En-tête
    st.markdown('<h1 class="main-header">💰 Simulateur d\'Impôt sur le Revenu Français</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Modélisation et simulation du calcul de l'impôt sur le revenu<br>
            Barème 2024 - Source: service-public.gouv.fr
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Onglets principaux
    tab1, tab2 = st.tabs(["🧮 Calculateur Individuel", "📊 Simulation Populationnelle"])
    
    with tab1:
        calculateur_individuel()
    
    with tab2:
        simulation_populationnelle()

def calculateur_individuel():
    """Onglet calculateur d'impôt individuel"""
    
    st.header("🧮 Calculateur d'Impôt Individuel")
    
    # Initialisation du calculateur
    calc = CalculateurImpot()
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("📋 Paramètres du Foyer")
        
        # Sélection du type de foyer
        type_foyer = st.selectbox(
            "Type de foyer fiscal",
            ["Célibataire", "Couple", "Famille monoparentale", "Personnalisé"]
        )
        
        # Paramètres selon le type de foyer
        if type_foyer == "Célibataire":
            nb_adultes = 1
            nb_enfants = 0
        elif type_foyer == "Couple":
            nb_adultes = 2
            nb_enfants = st.slider("Nombre d'enfants", 0, 8, 0)
        elif type_foyer == "Famille monoparentale":
            nb_adultes = 1
            nb_enfants = st.slider("Nombre d'enfants", 1, 8, 2)
        else:  # Personnalisé
            nb_adultes = st.selectbox("Nombre d'adultes", [1, 2])
            nb_enfants = st.slider("Nombre d'enfants", 0, 8, 0)
        
        # Revenu imposable
        revenu = st.number_input(
            "Revenu imposable (€)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
            format="%d"
        )
    
    # Calcul de l'impôt
    if st.button("Calculer l'impôt", type="primary"):
        resultat = calc.calculer_impot(revenu, nb_adultes, nb_enfants)
        
        # Affichage des résultats principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💳 Impôt à payer",
                value=f"{resultat['impot_final']:,.0f} €",
                delta=None
            )
        
        with col2:
            st.metric(
                label="📊 Taux moyen",
                value=f"{resultat['taux_moyen']*100:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                label="📈 Taux marginal",
                value=f"{resultat['taux_marginal']*100:.0f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                label="👥 Nombre de parts",
                value=f"{resultat['parts']:.1f}",
                delta=None
            )
        
        # Détail du calcul
        st.subheader("📋 Détail du Calcul")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informations générales:**")
            st.write(f"• Revenu total: {resultat['revenu_total']:,.0f} €")
            st.write(f"• Revenu par part: {resultat['revenu_par_part']:,.0f} €")
            st.write(f"• Impôt par part: {resultat['impot_par_part']:,.0f} €")
            
            st.markdown("**Étapes du calcul:**")
            st.write(f"• Impôt brut: {resultat['impot_brut']:,.0f} €")
            st.write(f"• Après décote: {resultat['impot_apres_decote']:,.0f} €")
            st.write(f"• **Impôt final: {resultat['impot_final']:,.0f} €**")
        
        with col2:
            st.markdown("**Détail par tranches:**")
            for i, tranche in enumerate(resultat['detail_tranches']):
                if tranche['montant_imposable'] > 0:
                    st.write(f"• {tranche['tranche']} ({tranche['taux']}): "
                           f"{tranche['montant_imposable']:,.0f}€ → {tranche['impot_tranche']:,.0f}€")
        
        # Visualisations
        st.subheader("📊 Visualisations")
        
        # Graphique du barème
        fig_bareme = go.Figure()
        
        # Barème progressif
        tranches = calc.bareme_2024['tranches']
        taux = calc.bareme_2024['taux']
        
        # Création des segments
        x_values = []
        y_values = []
        
        for i in range(len(tranches) - 1):
            x_values.extend([tranches[i], tranches[i+1]])
            y_values.extend([taux[i]*100, taux[i]*100])
        
        fig_bareme.add_trace(go.Scatter(
            x=x_values, y=y_values,
            mode='lines',
            name='Barème 2024',
            line=dict(color='blue', width=3)
        ))
        
        # Marqueur pour le revenu par part du contribuable
        fig_bareme.add_vline(
            x=resultat['revenu_par_part'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Votre revenu/part: {resultat['revenu_par_part']:,.0f}€"
        )
        
        fig_bareme.update_layout(
            title="Barème d'imposition 2024",
            xaxis_title="Revenu par part (€)",
            yaxis_title="Taux marginal (%)",
            height=400
        )
        
        st.plotly_chart(fig_bareme, use_container_width=True)
        
        # Graphique des taux moyen et marginal
        fig_taux = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Taux moyen selon le revenu", "Comparaison des taux")
        )
        
        # Taux moyen
        revenus = np.linspace(0, max(revenu*2, 100000), 1000)
        taux_moyens = []
        
        for r in revenus:
            if r > 0:
                impot = calc.calculer_impot_par_part(r)
                taux_moyen = impot / r * 100
                taux_moyens.append(taux_moyen)
            else:
                taux_moyens.append(0)
        
        fig_taux.add_trace(
            go.Scatter(x=revenus, y=taux_moyens, mode='lines', name='Taux moyen'),
            row=1, col=1
        )
        
        fig_taux.add_vline(
            x=resultat['revenu_par_part'],
            line_dash="dash",
            line_color="red",
            row=1, col=1
        )
        
        # Comparaison des taux
        fig_taux.add_trace(
            go.Bar(
                x=['Taux marginal', 'Taux moyen'],
                y=[resultat['taux_marginal']*100, resultat['taux_moyen']*100],
                name='Vos taux',
                marker_color=['orange', 'green']
            ),
            row=1, col=2
        )
        
        fig_taux.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_taux, use_container_width=True)
    
    # Section de comparaison de foyers
    st.subheader("🔍 Comparaison de Foyers")
    
    with st.expander("Comparer différents foyers fiscaux"):
        st.write("Sélectionnez plusieurs foyers pour comparer leur imposition:")
        
        col1, col2 = st.columns(2)
        
        foyers_comparaison = []
        
        with col1:
            st.markdown("**Foyer 1:**")
            r1 = st.number_input("Revenu 1 (€)", 0, 1000000, 30000, key="r1")
            a1 = st.selectbox("Adultes 1", [1, 2], key="a1")
            e1 = st.slider("Enfants 1", 0, 8, 0, key="e1")
            nom1 = st.text_input("Nom foyer 1", "Foyer 1", key="n1")
            
            st.markdown("**Foyer 2:**")
            r2 = st.number_input("Revenu 2 (€)", 0, 1000000, 60000, key="r2")
            a2 = st.selectbox("Adultes 2", [1, 2], key="a2")
            e2 = st.slider("Enfants 2", 0, 8, 0, key="e2")
            nom2 = st.text_input("Nom foyer 2", "Foyer 2", key="n2")
        
        with col2:
            st.markdown("**Foyer 3:**")
            r3 = st.number_input("Revenu 3 (€)", 0, 1000000, 80000, key="r3")
            a3 = st.selectbox("Adultes 3", [1, 2], key="a3")
            e3 = st.slider("Enfants 3", 0, 8, 2, key="e3")
            nom3 = st.text_input("Nom foyer 3", "Foyer 3", key="n3")
            
            st.markdown("**Foyer 4:**")
            r4 = st.number_input("Revenu 4 (€)", 0, 1000000, 45000, key="r4")
            a4 = st.selectbox("Adultes 4", [1, 2], key="a4")
            e4 = st.slider("Enfants 4", 0, 8, 2, key="e4")
            nom4 = st.text_input("Nom foyer 4", "Foyer 4", key="n4")
        
        if st.button("Comparer les foyers"):
            foyers = [
                {'nom': nom1, 'revenu': r1, 'nb_adultes': a1, 'nb_enfants': e1},
                {'nom': nom2, 'revenu': r2, 'nb_adultes': a2, 'nb_enfants': e2},
                {'nom': nom3, 'revenu': r3, 'nb_adultes': a3, 'nb_enfants': e3},
                {'nom': nom4, 'revenu': r4, 'nb_adultes': a4, 'nb_enfants': e4}
            ]
            
            resultats_comparaison = calc.comparer_foyers(foyers)
            
            # Tableau de comparaison
            df_comparaison = pd.DataFrame([{
                'Foyer': r['nom'],
                'Revenu (€)': f"{r['revenu']:,.0f}",
                'Parts': f"{r['parts']:.1f}",
                'Revenu/part (€)': f"{r['revenu_par_part']:,.0f}",
                'Impôt (€)': f"{r['impot_final']:,.0f}",
                'Taux moyen (%)': f"{r['taux_moyen']*100:.1f}",
                'Taux marginal (%)': f"{r['taux_marginal']*100:.0f}"
            } for r in resultats_comparaison])
            
            st.dataframe(df_comparaison, use_container_width=True)
            
            # Graphique de comparaison
            fig_comparaison = go.Figure()
            
            fig_comparaison.add_trace(go.Bar(
                x=[r['nom'] for r in resultats_comparaison],
                y=[r['impot_final'] for r in resultats_comparaison],
                name='Impôt total',
                marker_color='lightblue'
            ))
            
            fig_comparaison.update_layout(
                title="Comparaison de l'impôt entre foyers",
                xaxis_title="Foyer fiscal",
                yaxis_title="Impôt (€)",
                height=400
            )
            
            st.plotly_chart(fig_comparaison, use_container_width=True)

def simulation_populationnelle():
    """Onglet simulation populationnelle"""
    
    st.header("📊 Simulation Populationnelle")
    
    # Initialisation
    modele = ModelePopulationnel()
    sim = SimulationMacro()
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres de Simulation")
        
        # Mode de simulation
        mode = st.selectbox(
            "Mode de simulation",
            ["EDO (Déterministe)", "Markov (Stochastique)"]
        )
        
        # Durée de simulation
        duree = st.slider("Durée de simulation (années)", 5, 50, 20)
        
        # Paramètres de politique fiscale
        st.subheader("🎯 Politique Fiscale")
        
        delta_tau = st.slider(
            "Hausse du taux marginal supérieur",
            0.0, 0.15, 0.0, 0.01,
            help="Augmentation du taux de la tranche la plus élevée"
        )
        
        rho = st.slider(
            "Part redistribuée aux tranches basses",
            0.0, 0.3, 0.0, 0.01,
            help="Proportion redistribuée aux deux premières tranches"
        )
        
        # Paramètres économiques
        st.subheader("📈 Paramètres Économiques")
        
        croissance = st.slider("Croissance économique (%)", 0.0, 5.0, 2.0, 0.1)
        inflation = st.slider("Inflation (%)", 0.0, 5.0, 2.0, 0.1)
        
        # Mise à jour des paramètres du modèle
        modele.g = croissance / 100
        modele.pi = inflation / 100
        
        # Population
        population = st.number_input(
            "Population simulée",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000,
            format="%d"
        )
    
    # Bouton de simulation
    if st.button("🚀 Lancer la simulation", type="primary"):
        
        with st.spinner("Simulation en cours..."):
            # Simulation selon le mode choisi
            mode_sim = "edo" if mode == "EDO (Déterministe)" else "markov"
            
            if mode_sim == "edo":
                resultat = modele.simuler_edo(
                    T=duree, 
                    delta_tau=delta_tau, 
                    rho=rho, 
                    N_total=population
                )
            else:
                resultat = modele.simuler_markov(
                    T=duree, 
                    delta_tau=delta_tau, 
                    rho=rho, 
                    N_total=population
                )
            
            # Calcul des indicateurs avancés
            resultat = sim._calculer_indicateurs_avances(resultat)
            
            # Affichage des métriques principales
            st.subheader("📊 Résultats Principaux")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                recettes_evolution = (resultat['recettes_fiscales'][-1] - resultat['recettes_fiscales'][0]) / resultat['recettes_fiscales'][0] * 100
                st.metric(
                    label="💰 Recettes fiscales",
                    value=f"{resultat['recettes_fiscales'][-1]/1e6:.1f} M€",
                    delta=f"{recettes_evolution:+.1f}%"
                )
            
            with col2:
                gini_evolution = resultat['gini'][-1] - resultat['gini'][0]
                st.metric(
                    label="📊 Indice de Gini",
                    value=f"{resultat['gini'][-1]:.3f}",
                    delta=f"{gini_evolution:+.3f}"
                )
            
            with col3:
                taux_pauvrete_final = resultat['taux_pauvrete'][-1]
                st.metric(
                    label="👥 Taux de pauvreté",
                    value=f"{taux_pauvrete_final:.1f}%",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="🔄 Taux de mobilité",
                    value=f"{resultat['taux_mobilite']:.3f}",
                    delta=None
                )
            
            # Visualisations
            st.subheader("📈 Visualisations")
            
            # Graphique de l'évolution de la population
            fig_pop = go.Figure()
            
            for i in range(modele.k):
                fig_pop.add_trace(go.Scatter(
                    x=resultat['temps'],
                    y=resultat['population_par_tranche'][:, i],
                    mode='lines',
                    name=f'Tranche {i+1}',
                    line=dict(width=2)
                ))
            
            fig_pop.update_layout(
                title="Évolution de la population par tranche",
                xaxis_title="Temps (années)",
                yaxis_title="Population",
                height=400
            )
            
            st.plotly_chart(fig_pop, use_container_width=True)
            
            # Graphiques des métriques
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Recettes fiscales", "Indice de Gini", 
                              "Taux de pauvreté", "Revenus totaux"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Recettes fiscales
            fig_metrics.add_trace(
                go.Scatter(x=resultat['temps'], y=resultat['recettes_fiscales'],
                          mode='lines', name='Recettes', line=dict(color='green')),
                row=1, col=1
            )
            
            # Indice de Gini
            fig_metrics.add_trace(
                go.Scatter(x=resultat['temps'], y=resultat['gini'],
                          mode='lines', name='Gini', line=dict(color='red')),
                row=1, col=2
            )
            
            # Taux de pauvreté
            fig_metrics.add_trace(
                go.Scatter(x=resultat['temps'], y=resultat['taux_pauvrete'],
                          mode='lines', name='Pauvreté', line=dict(color='orange')),
                row=2, col=1
            )
            
            # Revenus totaux
            fig_metrics.add_trace(
                go.Scatter(x=resultat['temps'], y=[r/1e6 for r in resultat['revenus_totaux']],
                          mode='lines', name='Revenus', line=dict(color='blue')),
                row=2, col=2
            )
            
            fig_metrics.update_layout(height=600, showlegend=False)
            fig_metrics.update_yaxes(title_text="Recettes (€)", row=1, col=1)
            fig_metrics.update_yaxes(title_text="Gini", row=1, col=2)
            fig_metrics.update_yaxes(title_text="Pauvreté (%)", row=2, col=1)
            fig_metrics.update_yaxes(title_text="Revenus (M€)", row=2, col=2)
            
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Répartition finale
            st.subheader("🥧 Répartition Finale de la Population")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique en secteurs
                population_finale = resultat['population_par_tranche'][-1]
                labels = [f'Tranche {i+1}' for i in range(modele.k)]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=population_finale,
                    hole=0.3
                )])
                
                fig_pie.update_layout(title="Répartition finale", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Tableau détaillé
                df_repartition = pd.DataFrame({
                    'Tranche': labels,
                    'Population': population_finale,
                    'Pourcentage': [p/sum(population_finale)*100 for p in population_finale]
                })
                
                st.dataframe(df_repartition, use_container_width=True)
            
            # Analyse de sensibilité
            st.subheader("🔍 Analyse de Sensibilité")
            
            if st.checkbox("Effectuer une analyse de sensibilité"):
                with st.spinner("Analyse de sensibilité en cours..."):
                    # Test de différents scénarios
                    scenarios_sensibilite = [
                        {'nom': 'Status quo', 'delta_tau': 0, 'rho': 0},
                        {'nom': 'Hausse taux', 'delta_tau': delta_tau, 'rho': 0},
                        {'nom': 'Redistribution', 'delta_tau': 0, 'rho': rho},
                        {'nom': 'Réforme complète', 'delta_tau': delta_tau, 'rho': rho}
                    ]
                    
                    resultats_sensibilite = sim.analyser_impact_reforme(scenarios_sensibilite, duree)
                    
                    # Tableau de comparaison
                    tableau_sensibilite = sim.creer_tableau_synthese(resultats_sensibilite)
                    st.dataframe(tableau_sensibilite, use_container_width=True)
                    
                    # Graphique de comparaison
                    sim.visualiser_comparaison_scenarios(resultats_sensibilite)
    
    # Section d'informations
    with st.expander("ℹ️ À propos du modèle"):
        st.markdown("""
        **Modèle Populationnel:**
        
        - **EDO (Équations Différentielles Ordinaires)**: Modèle déterministe basé sur des flux entre tranches
        - **Markov (Chaîne de Markov)**: Modèle stochastique avec probabilités de transition
        
        **Paramètres:**
        
        - **Δτ (delta_tau)**: Hausse du taux marginal de la tranche supérieure
        - **ρ (rho)**: Part redistribuée aux tranches basses
        - **g**: Taux de croissance économique
        - **π**: Taux d'inflation
        
        **Indicateurs:**
        
        - **Recettes fiscales**: Total des impôts collectés
        - **Indice de Gini**: Mesure des inégalités (0 = égalité parfaite, 1 = inégalité maximale)
        - **Taux de pauvreté**: Proportion de la population dans la première tranche
        - **Taux de mobilité**: Capacité à changer de tranche d'imposition
        """)

if __name__ == "__main__":
    main()

