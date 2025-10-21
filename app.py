# ================================
# APPLICATION STREAMLIT COMPLÈTE
# Analyse de Risque de Crédit - VERSION CORRIGÉE
# ================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
import io
import shap
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page avec sidebar élargi
st.set_page_config(
    page_title="Analyse Risque Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour la sidebar BLEUE
st.markdown("""
<style>
    /* Sidebar bleue */
    [data-testid="stSidebar"] {
        background-color: #1e3c72;
        color: white;
    }
    
    /* Texte blanc dans la sidebar */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
            /* Boutons de navigation */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        padding: 0.75rem 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
        width: 100% !important;
        text-align: left !important;
        cursor: pointer !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        display: block !important;
    }
    .stButton > button:hover {
        background-color: #f0f0f0 !important;
        transform: translateX(5px) !important;
    }
    .stButton > button:focus {
        background-color: #f0f0f0 !important;
        border-left: 4px solid #ffd700 !important;
    }
    
    /* Texte des boutons */
    .stButton > button > div > p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Assurer que le texte reste noir dans tous les états */
    .stButton > button:hover > div > p {
        color: #000000 !important;
    }
    
    .stButton > button:focus > div > p {
        color: #000000 !important;
    }
    
    /* En-tête sidebar */
    .sidebar-header {
        color: white !important;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 2px solid #2e86ab;
    }
    
    /* Métriques dans la sidebar */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: white !important;
    }
    
    /* Style général pour le contenu principal */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Cartes métriques */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        margin-bottom: 1rem;
    }
    
    /* Groupes d'input */
    .input-group {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    /* Style pour les graphiques SHAP */
    .shap-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour ajouter les pourcentages aux graphiques
def ajouter_etiquettes_pourcentage(ax, total):
    """Ajoute des pourcentages aux barplots matplotlib"""
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 0.01
        ax.annotate(percentage, (x, y), ha='center', va='bottom')

@st.cache_data
def load_data():
    """Charge les données depuis le CSV avec gestion d'encodage"""
    try:
        # Essayer différents encodages
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv("data/base_analyse.csv", encoding=encoding)
#                st.success(f"✅ Données chargées avec encodage: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
        
        # Si aucun encodage ne fonctionne
        st.error("❌ Impossible de charger le fichier CSV avec les encodages standards")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Erreur de chargement des données: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Charge les modèles entraînés - VERSION RAPIDE"""
    models = {}
    try:
        st.info("🔄 Chargement du modèle XGBoost...")
        
        # Charger SEULEMENT XGBoost pour l'instant
        with open("assets/model_xgboost.pkl", 'rb') as f:
            models['XGBoost'] = joblib.load(f)
        
#        st.success("✅ Modèle XGBoost chargé!")
        return models
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        # Retourner un modèle vide pour éviter le blocage
        return {'XGBoost': None}

@st.cache_resource
def load_preprocessors():
    """Charge les preprocesseurs - VERSION RAPIDE"""
    try:
        with open("assets/label_encoder.pkl", 'rb') as f:
            le = joblib.load(f)
        with open("assets/scaler.pkl", 'rb') as f:
            scaler = joblib.load(f)
        return le, scaler
    except Exception as e:
        st.error(f"❌ Erreur preprocesseurs: {e}")
        return None, None

# ================================
# FONCTIONS POUR L'ANALYSE SHAP
# ================================

def compute_shap_analysis(model, input_data, feature_names, model_name):
    """Calcule et retourne les valeurs SHAP pour l'explication des prédictions"""
    try:
        # Vérifier que le modèle supporte predict_proba
        if not hasattr(model, 'predict_proba'):
            return None, None
            
        # Convertir en numpy array pour éviter les problèmes d'encodage
        input_array = input_data.values if hasattr(input_data, 'values') else input_data
        
        # Initialiser l'explainer SHAP selon le type de modèle
        if model_name in ['XGBoost', 'Random Forest', 'Decision Tree']:
            # Pour XGBoost, utiliser TreeExplainer avec paramètres spécifiques
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="probability"
            )
            shap_values = explainer.shap_values(input_array)
            
            # Pour les classifieurs, shap_values peut être une liste [classe_0, classe_1]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # Prendre les valeurs pour la classe positive (DEFAUT)
            elif isinstance(shap_values, list) and len(shap_values) > 2:
                shap_values = shap_values[-1]  # Dernière classe
                
        elif model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(model, input_array)
            shap_values = explainer.shap_values(input_array)
        else:
            # Pour SVM et autres modèles, utiliser KernelExplainer avec moins d'échantillons
            explainer = shap.KernelExplainer(
                model.predict_proba, 
                shap.sample(input_array, min(100, len(input_array)))
            )
            shap_values = explainer.shap_values(input_array, nsamples=100)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
        
        # S'assurer que shap_values est un array 2D
        if shap_values is not None:
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Vérifier que la forme correspond aux features
            if shap_values.shape[1] != len(feature_names):
                st.warning(f"Incompatibilité: {shap_values.shape[1]} valeurs SHAP vs {len(feature_names)} features")
                return None, None
        
        return explainer, shap_values
        
    except Exception as e:
        st.warning(f"Analyse SHAP non disponible: {str(e)}")
        return None, None

def plot_shap_summary(shap_values, input_data, feature_names):
    """Crée un graphique summary plot SHAP interactif"""
    if shap_values is None:
        return None
    
    # Créer un DataFrame pour les valeurs SHAP
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values[0],
        'feature_value': input_data.values[0]
    })
    
    # Trier par valeur SHAP absolue
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=False)
    
    # Créer le graphique à barres
    fig = px.bar(
        shap_df.head(10),
        x='shap_value',
        y='feature',
        orientation='h',
        title='📊 Impact des Variables sur la Prédiction (Top 10)',
        labels={'shap_value': 'Impact SHAP', 'feature': 'Variables'},
        color='shap_value',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500
    )
    
    return fig

def create_shap_detailed_table(shap_values, input_data, feature_names):
    """Crée un tableau détaillé des contributions SHAP"""
    if shap_values is None:
        return None
    
    # Créer le DataFrame détaillé
    shap_details = pd.DataFrame({
        'Variable': feature_names,
        'Valeur': input_data.values[0],
        'Impact SHAP': shap_values[0],
        'Impact Absolu': np.abs(shap_values[0])
    })
    
    # Trier par impact absolu
    shap_details = shap_details.sort_values('Impact Absolu', ascending=False)
    
    # Ajouter une colonne d'interprétation
    def get_interpretation(row):
        if row['Impact SHAP'] < 0:
            return f"Augmente le risque de {row['Impact SHAP']:.4f}"
        else:
            return f"Réduit le risque de {abs(row['Impact SHAP']):.4f}"
    
    shap_details['Interprétation'] = shap_details.apply(get_interpretation, axis=1)
    
    return shap_details

# ================================
# NAVIGATION VERTICALE
# ================================

def show_sidebar_navigation():
    """Affiche la navigation verticale dans la sidebar"""
    st.sidebar.markdown('<div class="sidebar-header">🏦 Analyse Risque Crédit</div>', unsafe_allow_html=True)
    
    # Boutons de navigation avec texte blanc visible
    if st.sidebar.button("🏠 Accueil", use_container_width=True, key="Accueil"):
        st.session_state.page = "Accueil"
    
    if st.sidebar.button("🔍 Analyse Exploratoire", use_container_width=True, key="Analyse Exploratoire"):
        st.session_state.page = "Analyse Exploratoire"
    
    if st.sidebar.button("📈 Statistiques Descriptives", use_container_width=True, key="Statistiques Descriptives"):
        st.session_state.page = "Statistiques Descriptives"
    
    if st.sidebar.button("🤖 Performance des Modèles", use_container_width=True, key="Performance des Modèles"):
        st.session_state.page = "Performance des Modèles"
    
    if st.sidebar.button("🔮 Prédictions", use_container_width=True, key="Prédictions"):
        st.session_state.page = "Prédictions"
    
    # Séparateur
    st.sidebar.markdown("---")
    
    # Informations de statut
    try:
        df = load_data()
        st.sidebar.markdown("**📊 Aperçu du Portefeuille**")
        st.sidebar.metric("Total Crédits", f"{len(df):,}")
        
        # Trouver la colonne de statut
        status_column = None
        for col in ['statut_actuel', 'statut', 'target', 'defaut']:
            if col in df.columns:
                status_column = col
                break
        
        if status_column:
            if df[status_column].dtype == 'object':
                defaut_rate = (df[status_column] == 'DEFAUT').mean() * 100
            else:
                defaut_rate = (df[status_column] == 1).mean() * 100
            st.sidebar.metric("Taux de Défaut", f"{defaut_rate:.2f}%")
            
    except Exception as e:
        st.sidebar.info("Chargement des données...")

# ================================
# FONCTIONS D'AFFICHAGE DES PAGES
# ================================

def show_home():
    """Page d'accueil"""
    st.markdown('<div class="main-header">🏦 Analyse de Risque de Crédit</div>', unsafe_allow_html=True)
    
    # Bannière d'accueil
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin: 0; font-size: 2.5rem;'>Plateforme d'Analyse Prédictive Avancée</h2>
        <p style='font-size: 1.2rem; opacity: 0.9;'>Outils avancés pour l'évaluation du risque de crédit avec explications SHAP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cartes de fonctionnalités
    st.markdown('<div class="section-header"> Fonctionnalités Principales</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3> Analyse Exploratoire</h3>
            <p>Exploration complète des données avec visualisations interactives et tableaux statistiques détaillés.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3> Modélisation Avancée</h3>
            <p>Algorithmes de machine learning (XGBoost, Random Forest, etc.) pour une prédiction précise du risque.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3> Prédictions Intelligentes</h3>
            <p>Interface intuitive avec explications SHAP pour comprendre les décisions du modèle.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3> Analyse SHAP</h3>
            <p>Compréhension détaillée de l'impact de chaque variable sur les décisions de crédit.</p>
        </div>
        """, unsafe_allow_html=True)

def show_exploratory_analysis():
    """Analyse exploratoire complète"""
    st.markdown('<div class="section-header">🔍 Analyse Exploratoire Complète</div>', unsafe_allow_html=True)
    
    try:
        df = load_data()
        
        # Onglets pour organiser l'analyse
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Tableau Statistique", "📊 Répartitions", "📉 Distributions", 
            "🔗 Corrélations", "📅 Séries Temporelles"
        ])
        
        with tab1:
            show_statistical_table(df)
            
        with tab2:
            show_distributions(df)
            
        with tab3:
            show_variable_distributions(df)
            
        with tab4:
            show_correlation_analysis(df)
            
        with tab5:
            show_temporal_analysis(df)
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse exploratoire: {e}")

def show_statistical_table(df):
    """Affiche le tableau statistique complet"""
    st.subheader("📊 Tableau Statistique des Variables Quantitatives")
    
    # Sélection des variables quantitatives
    quant_vars = [
        "capital_paye_t3", "interet_paye_t3", "nb_ech_payees_t3", 
        "retard_cumule_jours_t3", "retard_moyen_jours_t3", 
        "nb_mvts_credit_t3", "nb_mvts_debit_t3", 
        "montant_credit_total_t3", "montant_debit_total_t3", 
        "montant_moy_t3",  
        "pct_capital_paye_t3", "ecart_prev_reel_t3", 
        "ratio_interet_sur_capital_paye_t3", "pct_interet_paye_t3", 
        "retard_cumule_ratio_t3", "ratio_debit_credit_t3", 
        "duree_ecart_t3", "pct_retards_t3"
    ]
    
    # Vérifier quelles variables existent dans le dataframe
    available_vars = [var for var in quant_vars if var in df.columns]
    
    if available_vars:
        # Création du tableau statistique
        tableau_stats = df[available_vars].describe().T
        
        # Ajout de statistiques supplémentaires
        tableau_stats['median'] = df[available_vars].median()
        tableau_stats['iqr'] = tableau_stats['75%'] - tableau_stats['25%']
        
        # Renommage des colonnes
        tableau_stats = tableau_stats.rename(columns={
            'count': 'Effectif',
            'mean': 'Moyenne',
            'std': 'Écart-type',
            'min': 'Minimum',
            '25%': 'Q1',
            '50%': 'Médiane',
            '75%': 'Q3',
            'max': 'Maximum',
            'iqr': 'IQR'
        })
        
        # Réorganisation des colonnes
        tableau_stats = tableau_stats[[
            'Effectif', 'Moyenne', 'Écart-type', 'Minimum', 'Q1', 
            'Médiane', 'Q3', 'Maximum', 'IQR'
        ]]
        
        # Formater le tableau
        formatted_table = tableau_stats.copy()
        for col in ['Moyenne', 'Écart-type', 'Minimum', 'Q1', 'Médiane', 'Q3', 'Maximum', 'IQR']:
            if col in formatted_table.columns:
                formatted_table[col] = formatted_table[col].round(2)
        
        # Affichage avec style
        st.dataframe(formatted_table.style.background_gradient(cmap='Blues'))
        
        # Téléchargement
        csv = tableau_stats.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Télécharger le tableau statistique (CSV)",
            data=csv,
            file_name="tableau_statistique_credits.csv",
            mime="text/csv"
        )
    else:
        st.warning("Aucune variable quantitative trouvée dans le dataset")

def show_distributions(df):
    """Affiche les répartitions catégorielles"""
    st.subheader("📊 Répartitions des Variables Catégorielles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition par statut
        status_column = None
        for col in ['statut_actuel', 'statut', 'target', 'defaut']:
            if col in df.columns:
                status_column = col
                break
        
        if status_column:
            statut_counts = df[status_column].value_counts()
            fig = px.pie(
                values=statut_counts.values,
                names=statut_counts.index,
                title=f'Répartition par {status_column}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Colonne de statut non trouvée")
        
        # Répartition par type de crédit
        if 'credit_type' in df.columns:
            credit_counts = df['credit_type'].value_counts()
            fig = px.bar(
                x=credit_counts.index,
                y=credit_counts.values,
                title='Répartition par Type de Crédit',
                labels={'x': 'Type de crédit', 'y': 'Nombre'},
                color=credit_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Répartition par type de client
        if 'client_type' in df.columns:
            client_counts = df['client_type'].value_counts()
            fig = px.pie(
                values=client_counts.values,
                names=client_counts.index,
                title='Répartition par Type de Client'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Répartition par région
        if 'region' in df.columns:
            region_data = df[df['region'].str.lower() != 'inconnu'] if 'region' in df.columns else df
            if not region_data.empty and 'region' in region_data.columns:
                region_counts = region_data['region'].value_counts().head(10)  # Top 10
                fig = px.bar(
                    x=region_counts.index,
                    y=region_counts.values,
                    title='Top 10 Régions',
                    labels={'x': 'Région', 'y': 'Nombre'},
                    color=region_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

def show_variable_distributions(df):
    """Affiche les distributions des variables quantitatives"""
    st.subheader("📉 Distributions des Variables Quantitatives")
    
    # Variables pour histogrammes
    hist_vars = ["capital_paye_t3", "retard_cumule_jours_t3", "ratio_debit_credit_t3"]
    available_hist_vars = [var for var in hist_vars if var in df.columns]
    
    if available_hist_vars:
        col1, col2 = st.columns(2)
        
        with col1:
            for var in available_hist_vars[:2]:
                fig = px.histogram(
                    df, 
                    x=var, 
                    nbins=30,
                    title=f'Distribution de {var}',
                    marginal='box',
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            for var in available_hist_vars[2:]:
                fig = px.histogram(
                    df, 
                    x=var, 
                    nbins=30,
                    title=f'Distribution de {var}',
                    marginal='box',
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Boxplots par statut
    st.subheader("📦 Analyse par Statut")
    
    box_vars = ["pct_capital_paye_t3", "retard_moyen_jours_t3", "nb_mvts_debit_t3", "nb_mvts_credit_t3", "pct_retards_t3"]
    available_box_vars = [var for var in box_vars if var in df.columns]
    
    # Trouver la colonne de statut
    status_column = None
    for col in ['statut_actuel', 'statut', 'target', 'defaut']:
        if col in df.columns:
            status_column = col
            break
    
    if available_box_vars and status_column:
        selected_var = st.selectbox("Choisir une variable pour l'analyse:", available_box_vars)
        
        fig = px.box(
            df, 
            x=status_column, 
            y=selected_var,
            title=f'{selected_var} par {status_column}',
            color=status_column
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données insuffisantes pour l'analyse par statut")

def show_correlation_analysis(df):
    """Affiche l'analyse de corrélation"""
    st.subheader("🔗 Analyse des Corrélations")
    
    # Sélectionner les variables numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        # Matrice de corrélation
        corr_matrix = numeric_df.corr()
        
        # Heatmap interactive
        fig = px.imshow(
            corr_matrix,
            title="Matrice de Corrélation - Variables Numériques",
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top des corrélations
        st.subheader("Top des Corrélations")
        
        # Créer un dataframe des corrélations
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        # Supprimer les auto-corrélations et trier par valeur absolue
        corr_pairs = corr_pairs[corr_pairs != 1.0]
        corr_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)
        
        # Afficher les top 15
        top_correlations = corr_pairs.head(15)
        corr_df = top_correlations.reset_index()
        corr_df.columns = ['Variable 1', 'Variable 2', 'Corrélation']
        
        # Style the correlation table
        st.dataframe(corr_df.style.background_gradient(
            subset=['Corrélation'], 
            cmap='RdBu_r', 
            vmin=-1, 
            vmax=1
        ).format({'Corrélation': '{:.3f}'}), use_container_width=True)
        
    else:
        st.warning("Pas assez de variables numériques pour l'analyse de corrélation")

def show_temporal_analysis(df):
    """Affiche l'analyse temporelle"""
    st.subheader("📅 Analyse Temporelle")
    
    # Vérifier si la colonne date existe
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    
    if date_columns:
        selected_date_col = st.selectbox("Choisir la colonne de date:", date_columns)
        
        try:
            # Conversion en datetime
            df_temp = df.copy()
            df_temp[selected_date_col] = pd.to_datetime(df_temp[selected_date_col], errors='coerce')
            
            # Nettoyer les dates invalides
            df_temp = df_temp.dropna(subset=[selected_date_col])
            
            if not df_temp.empty:
                # Extraire année et mois
                df_temp['annee'] = df_temp[selected_date_col].dt.year
                df_temp['mois'] = df_temp[selected_date_col].dt.month
                
                # Agrégation par mois et année
                credits_par_mois = df_temp.groupby(['annee', 'mois']).size().reset_index(name='nb_credits')
                
                if not credits_par_mois.empty:
                    # Créer une colonne date pour l'axe x
                    credits_par_mois['date'] = pd.to_datetime(
                        credits_par_mois['annee'].astype(str) + '-' + credits_par_mois['mois'].astype(str) + '-01'
                    )
                    
                    # Courbe temporelle
                    fig = px.line(
                        credits_par_mois.sort_values('date'),
                        x='date',
                        y='nb_credits',
                        title='Évolution des Octrois de Crédits',
                        markers=True,
                        labels={'date': 'Date', 'nb_credits': 'Nombre de crédits'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques temporelles
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Période couverte", f"{credits_par_mois['annee'].min()} - {credits_par_mois['annee'].max()}")
                    with col2:
                        st.metric("Mois le plus actif", f"{credits_par_mois.loc[credits_par_mois['nb_credits'].idxmax(), 'date'].strftime('%b %Y')}")
                    with col3:
                        st.metric("Crédits total", f"{credits_par_mois['nb_credits'].sum():,}")
                else:
                    st.warning("Aucune donnée temporelle valide trouvée")
                    
        except Exception as e:
            st.warning(f"Impossible d'analyser les dates: {e}")
    else:
        st.info("Aucune colonne de date trouvée pour l'analyse temporelle")

def show_descriptive_stats():
    """Statistiques descriptives de base"""
    st.markdown('<div class="section-header">📈 Statistiques Descriptives</div>', unsafe_allow_html=True)
    
    try:
        df = load_data()
        
        # KPIs principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_credits = len(df)
            st.metric("Total Crédits", f"{total_credits:,}")
        
        with col2:
            # Trouver la colonne de statut
            status_column = None
            for col in ['statut_actuel', 'statut', 'target', 'defaut']:
                if col in df.columns:
                    status_column = col
                    break
            
            if status_column:
                if df[status_column].dtype == 'object':
                    defaut_rate = (df[status_column] == 'DEFAUT').mean() * 100
                else:
                    defaut_rate = (df[status_column] == 1).mean() * 100
                st.metric("Taux de Défaut", f"{defaut_rate:.2f}%")
            else:
                st.metric("Variables", len(df.columns))
        
        with col3:
            st.metric("Variables", len(df.columns))
        
        with col4:
            if 'capital_paye_t3' in df.columns:
                avg_capital = df['capital_paye_t3'].mean()
                st.metric("Capital Moyen Payé", f"{avg_capital:,.2f}XOF")
            else:
                st.metric("Première colonne", df.columns[0])
        
        # Graphiques de base
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des statuts si disponible
            status_column = None
            for col in ['statut_actuel', 'statut', 'target', 'defaut']:
                if col in df.columns:
                    status_column = col
                    break
            
            if status_column:
                status_counts = df[status_column].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title=f'Distribution des {status_column}'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune colonne de statut trouvée")
        
        with col2:
            # Histogramme d'une variable numérique
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_num_col = st.selectbox("Choisir une variable numérique:", numeric_cols)
                fig = px.histogram(
                    df, 
                    x=selected_num_col, 
                    title=f'Distribution de {selected_num_col}',
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Aperçu des données
        st.subheader("Aperçu des Données")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Informations sur le dataset
        with st.expander("📋 Informations du Dataset"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Forme du dataset:**", df.shape)
                st.write("**Colonnes numériques:**", len(df.select_dtypes(include=[np.number]).columns))
                st.write("**Colonnes catégorielles:**", len(df.select_dtypes(include=['object']).columns))
            with col2:
                st.write("**Valeurs manquantes:**", df.isnull().sum().sum())
                st.write("**Doublons:**", df.duplicated().sum())
                
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")

def show_model_performance():
    """Affiche les performances des modèles"""
    st.markdown('<div class="section-header">🤖 Performance des Modèles</div>', unsafe_allow_html=True)
    
    try:
        # Charger les résultats depuis le fichier Excel
        results_df = pd.read_excel("assets/resultats_modeles.xlsx", index_col=0)
        
        # Métriques principales
        st.subheader("Comparaison des Modèles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_auc_model = results_df.loc[results_df['AUC'].idxmax()]
            st.metric("Meilleur AUC", f"{best_auc_model['AUC']:.4f}", best_auc_model.name)
        
        with col2:
            best_acc_model = results_df.loc[results_df['Accuracy'].idxmax()]
            st.metric("Meilleure Accuracy", f"{best_acc_model['Accuracy']:.4f}", best_acc_model.name)
        
        with col3:
            best_cv_model = results_df.loc[results_df['CV_mean'].idxmax()]
            st.metric("Meilleure CV Score", f"{best_cv_model['CV_mean']:.4f}", best_cv_model.name)
        
        # Graphiques de comparaison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                results_df.reset_index(),
                x='index',
                y='AUC',
                title='AUC par Modèle',
                color='AUC',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_title="Modèle", yaxis_title="AUC")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=results_df.index, y=results_df['Accuracy']))
            fig.add_trace(go.Bar(name='CV Score', x=results_df.index, y=results_df['CV_mean']))
            fig.update_layout(
                title='Accuracy vs Cross-Validation Score',
                xaxis_title="Modèle",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.subheader("Tableau Détaillé des Performances")
        
        # Formater le tableau
        styled_df = results_df.style.format({
            'Accuracy': '{:.4f}',
            'AUC': '{:.4f}', 
            'CV_mean': '{:.4f}',
            'CV_std': '{:.4f}'
        }).background_gradient(subset=['AUC', 'Accuracy'], cmap='Blues')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Téléchargement des résultats
        csv = results_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv,
            file_name="resultats_modeles.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des résultats: {e}")
        st.info("Assurez-vous que le fichier 'assets/resultats_modeles.xlsx' existe")

# def show_predictions():
#     """Interface de prédictions avec analyse SHAP"""
#     st.markdown('<div class="section-header">🔮 Prédictions en Temps Réel avec Analyse SHAP</div>', unsafe_allow_html=True)
    
#     models = load_models()
#     le, scaler = load_preprocessors()
    
#     if not models:
#         st.error("❌ Impossible de charger les modèles")
#         st.info("Veuillez vérifier que les fichiers .pkl sont dans le dossier 'assets/'")
#         return
    
#     if le is None:
#         st.error("❌ Impossible de charger le label encoder")
#         return
    
#     # Dictionnaires pour les variables qualitatives
#     client_type_options = {
#         1: "Particulier",
#         2: "Entreprise"
#     }
    
#     credit_type_options = {
#         1: "Crédit par signature",
#         2: "Crédit de caisse", 
#         3: "Crédit immobilier",
#         4: "Financement équipement",
#         5: "Avance sur marché",
#         6: "Financement fonds de commerce",
#         7: "Microcrédit",
#         8: "Crédit à la consommation"
#     }
    
#     region_options = {
#         1: "OUEST",
#         2: "NORD",
#         3: "SUD-OUEST", 
#         4: "SUD-OUEST",
#         5: "NORD-EST",
#         6: "SUD",
#         7: "CENTRE",
#         8: "CENTRE-OUEST",
#         9: "CENTRE-EST",
#         10: "CENTRE-NORD",
#         11: "CENTRE-SUD"
#     }
    
#     sector_options = {
#         1: "Commerce",
#         2: "BTP",
#         3: "Industrie",
#         4: "Services",
#         5: "Agriculture", 
#         6: "Transport",
#         7: "Restauration",
#         8: "Immobilier",
#         9: "Finance",
#         10: "Santé",
#         11: "Education",
#         12: "IT",
#         13: "Artisanat",
#         14: "Energie",
#         15: "Autres"
#     }
    
#     # Interface de saisie avec design amélioré
#     st.markdown("""
#     <div class="input-group">
#         <h3>📋 Saisie des Caractéristiques du Crédit</h3>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Organisation des champs en colonnes
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("**💰 Variables Financières**")
#         capital_paye = st.number_input("Capital Payé (t3)", min_value=0.0, value=50000.0, step=1000.0, format="%.4f")
#         interet_paye = st.number_input("Intérêts Payés (t3)", min_value=0.0, value=5000.0, step=100.0, format="%.4f")
#         montant_credit = st.number_input("Montant Crédit Total (t3)", min_value=0.0, value=100000.0, step=1000.0, format="%.4f")
#         montant_debit = st.number_input("Montant Débit Total (t3)", min_value=0.0, value=80000.0, step=1000.0, format="%.4f")
#         montant_moy = st.number_input("Montant Moyen (t3)", min_value=0.0, value=5000.0, step=100.0, format="%.4f")
        
#     with col2:
#         st.markdown("**📊 Ratios et Pourcentages**")
#         ratio_interet = st.number_input("Ratio Intérêt/Capital (t3)", min_value=0.0, value=0.1, step=0.0001, format="%.4f")
#         pct_interet = st.number_input("Pourcentage Intérêt Payé (t3)", min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")
#         pct_capital = st.number_input("Pourcentage Capital Payé (t3)", min_value=0.0, max_value=1.0, value=0.6, step=0.0001, format="%.4f")
#         retard_cumule_ratio = st.number_input("Ratio Retard Cumulé (t3)", min_value=0.0, value=0.05, step=0.0001, format="%.4f")
#         ratio_debit_credit = st.number_input("Ratio Débit/Crédit (t3)", min_value=0.0, value=0.8, step=0.0001, format="%.4f")
#         pct_retards = st.number_input("Pourcentage Retards (t3)", min_value=0.0, max_value=1.0, value=0.1, step=0.0001, format="%.4f")
        
#     with col3:
#         st.markdown("**🔢 Comportement de Paiement**")
#         nb_ech_payees = st.number_input("Nb Échéances Payées (t3)", min_value=0, value=12, step=1)
#         retard_moyen = st.number_input("Retard Moyen (jours t3)", min_value=0.0, value=5.0, step=0.1, format="%.1f")
#         nb_mvts_credit = st.number_input("Nb Mouvements Crédit (t3)", min_value=0, value=10, step=1)
#         nb_mvts_debit = st.number_input("Nb Mouvements Débit (t3)", min_value=0, value=8, step=1)
#         duree_ecart = st.number_input("Durée Écart (t3)", min_value=0, value=30, step=1)
#          # Ajout des champs timestamp si nécessaire
#         st.markdown("**📅 Dates des Mouvements**")
#         premier_mouvement_month = st.number_input("Mois Premier Mouvement (t3)", min_value=1, max_value=12, value=1)
#         premier_mouvement_day = st.number_input("Jour Premier Mouvement (t3)", min_value=1, max_value=31, value=1)
#         dernier_mouvement_month = st.number_input("Mois Dernier Mouvement (t3)", min_value=1, max_value=12, value=2)
#         dernier_mouvement_day = st.number_input("Jour Dernier Mouvement (t3)", min_value=1, max_value=31, value=1)
        
#         st.markdown("**👥 Informations Client**")
#         client_type = st.selectbox("Type de Client", options=list(client_type_options.keys()), 
#                                  format_func=lambda x: client_type_options[x])
#         credit_type = st.selectbox("Type de Crédit", options=list(credit_type_options.keys()),
#                                  format_func=lambda x: credit_type_options[x])
#         region = st.selectbox("Région", options=list(region_options.keys()),
#                             format_func=lambda x: region_options[x])
#         sector = st.selectbox("Secteur d'Activité", options=list(sector_options.keys()),
#                             format_func=lambda x: sector_options[x])
    
#         # Utilisation d'un seul modèle (XGBoost)
#     selected_model = 'XGBoost'
#     st.markdown("""
#     <div class="input-group">
#         <h3>🤖 Modèle Utilisé</h3>
#     </div>
#     """, unsafe_allow_html=True)

#     st.info(f"**Modèle utilisé :** {selected_model}")

#     # Option pour l'analyse SHAP
#     st.markdown("""
#     <div class="input-group">
#         <h3>🔍 Analyse Explicative SHAP</h3>
#     </div>
#     """, unsafe_allow_html=True)
        
#     # Option pour l'analyse SHAP
#     st.markdown("""
#     <div class="input-group">
#         <h3>🔍 Analyse Explicative SHAP</h3>
#     </div>
#     """, unsafe_allow_html=True)
    
#     enable_shap = st.checkbox("Activer l'analyse SHAP détaillée", value=True,
#                             help="Afficher l'analyse détaillée de l'impact de chaque variable sur la prédiction")
    
#     # Bouton de prédiction stylisé
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         predict_button = st.button("🚀 Lancer la Prédiction et l'Analyse SHAP", 
#                                  use_container_width=True, 
#                                  type="primary",
#                                  help="Cliquez pour analyser le risque de défaut avec explications détaillées")
    
#     if predict_button:
#         # ORDRE EXACT des colonnes utilisé pendant l'entraînement
#         # ORDRE EXACT des colonnes utilisé pendant l'entraînement (avec les timestamps)
#         expected_columns = [
#             'capital_paye_t3', 'interet_paye_t3', 'nb_ech_payees_t3', 
#             'retard_moyen_jours_t3', 'nb_mvts_credit_t3', 'nb_mvts_debit_t3', 
#             'montant_credit_total_t3', 'montant_debit_total_t3', 'ratio_interet_sur_capital_paye_t3', 
#             'pct_interet_paye_t3', 'montant_moy_t3', 'pct_capital_paye_t3', 
#             'retard_cumule_ratio_t3', 'ratio_debit_credit_t3', 'duree_ecart_t3', 
#             'pct_retards_t3', 'premier_mouvement_t3_timestamp', 'premier_mouvement_t3_month', 
#             'premier_mouvement_t3_day', 'dernier_mouvement_t3_timestamp', 
#             'dernier_mouvement_t3_month', 'dernier_mouvement_t3_day', 
#             'client_type', 'credit_type', 'region', 'sector'
#         ]
        
#         # Préparation des features avec l'ORDRE EXACT (incluant les timestamps)
#         # Pour les timestamps, on utilise des valeurs par défaut réalistes
#         input_data = pd.DataFrame({
#             'capital_paye_t3': [capital_paye],
#             'interet_paye_t3': [interet_paye],
#             'nb_ech_payees_t3': [nb_ech_payees],
#             'retard_moyen_jours_t3': [retard_moyen],
#             'nb_mvts_credit_t3': [nb_mvts_credit],
#             'nb_mvts_debit_t3': [nb_mvts_debit],
#             'montant_credit_total_t3': [montant_credit],
#             'montant_debit_total_t3': [montant_debit],
#             'ratio_interet_sur_capital_paye_t3': [ratio_interet],
#             'pct_interet_paye_t3': [pct_interet],
#             'montant_moy_t3': [montant_moy],
#             'pct_capital_paye_t3': [pct_capital],
#             'retard_cumule_ratio_t3': [retard_cumule_ratio],
#             'ratio_debit_credit_t3': [ratio_debit_credit],
#             'duree_ecart_t3': [duree_ecart],
#             'pct_retards_t3': [pct_retards],
#             # Ajout des colonnes timestamp avec des valeurs par défaut
#             'premier_mouvement_t3_timestamp': [1640995200],  # 1er janvier 2022
#             'premier_mouvement_t3_month': [1],
#             'premier_mouvement_t3_day': [1],
#             'dernier_mouvement_t3_timestamp': [1643673600],  # 1er février 2022
#             'dernier_mouvement_t3_month': [2],
#             'dernier_mouvement_t3_day': [1],
#             'client_type': [client_type],
#             'credit_type': [credit_type],
#             'region': [region],
#             'sector': [sector]
#         })
        
#         # Réorganiser les colonnes dans l'ordre exact utilisé pendant l'entraînement
#         input_data = input_data[expected_columns]
        
#                 # Prédiction avec XGBoost uniquement
#         model = models['XGBoost']

#         # Vérifier si c'est un dictionnaire et extraire le modèle
#         if isinstance(model, dict):
#             # Si c'est un dictionnaire, extraire le modèle
#             if 'model' in model:
#                 model = model['model']
#             elif 'classifier' in model:
#                 model = model['classifier']
#             elif 'estimator' in model:
#                 model = model['estimator']
#             else:
#                 # Prendre le premier élément du dictionnaire
#                 model = list(model.values())[0]

#         try:
#             # Pour XGBoost, pas besoin de scaler spécifique
#             prediction_encoded = model.predict(input_data)[0]
#             proba = model.predict_proba(input_data)[0]
            
#             prediction = le.inverse_transform([prediction_encoded])[0]
            
#             # Affichage des résultats avec design amélioré
#             st.markdown("---")
#             st.markdown('<div class="section-header">🎯 Résultat de la Prédiction</div>', unsafe_allow_html=True)
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 if prediction == 'SOLDE':
#                     st.success(f"## ✅ CRÉDIT SOLDE")
#                     st.metric(
#                         "Probabilité de Solde", 
#                         f"{proba[1]:.4%}",
#                         delta=f"Risque faible: {proba[0]:.4%}" 
#                     )
#                 else:
#                     st.error(f"## 🚨 RISQUE DE DÉFAUT")
#                     st.metric(
#                         "Probabilité de Défaut", 
#                         f"{proba[0]:.4%}",
#                         delta=f"Solde: {proba[1]:.4%}",
#                         delta_color="inverse"
#                     )
                    
#             with col2:
#                 # Jauge de risque améliorée
#                 risk_score = proba[0] * 100
#                 fig = go.Figure(go.Indicator(
#                     mode = "gauge+number+delta",
#                     value = risk_score,
#                     domain = {'x': [0, 1], 'y': [0, 1]},
#                     title = {'text': "Score de Risque", 'font': {'size': 20}},
#                     delta = {'reference': 50},
#                     gauge = {
#                         'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
#                         'bar': {'color': "darkblue"},
#                         'bgcolor': "white",
#                         'borderwidth': 2,
#                         'bordercolor': "gray",
#                         'steps': [
#                             {'range': [0, 20], 'color': "lightgreen"},
#                             {'range': [20, 50], 'color': "yellow"},
#                             {'range': [50, 80], 'color': "orange"},
#                             {'range': [80, 100], 'color': "red"}
#                         ],
#                         'threshold': {
#                             'line': {'color': "red", 'width': 4},
#                             'thickness': 0.75,
#                             'value': 90
#                         }
#                     }
#                 ))
#                 fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
#                 st.plotly_chart(fig, use_container_width=True)

#         except Exception as e:
#             st.error(f"❌ Erreur lors de la prédiction: {e}")
#             st.info("Vérifiez que toutes les features nécessaires sont fournies")
#             # Arrêter l'exécution ici pour éviter d'autres erreurs
#             st.stop()
def show_predictions():
    """Interface de prédictions avec analyse SHAP"""
    st.markdown('<div class="section-header">🔮 Prédictions en Temps Réel avec Analyse SHAP</div>', unsafe_allow_html=True)
    
    models = load_models()
    le, scaler = load_preprocessors()
    
    if not models:
        st.error("❌ Impossible de charger les modèles")
        return
    
    if le is None:
        st.error("❌ Impossible de charger le label encoder")
        return
    
    # Dictionnaires pour les variables qualitatives
    client_type_options = {1: "Particulier", 2: "Entreprise"}
    credit_type_options = {
        1: "Crédit par signature", 2: "Crédit de caisse", 3: "Crédit immobilier",
        4: "Financement équipement", 5: "Avance sur marché", 6: "Financement fonds de commerce",
        7: "Microcrédit", 8: "Crédit à la consommation"
    }
    region_options = {
        1: "OUEST", 2: "NORD", 3: "SUD-OUEST", 4: "SUD-OUEST", 5: "NORD-EST",
        6: "SUD", 7: "CENTRE", 8: "CENTRE-OUEST", 9: "CENTRE-EST", 10: "CENTRE-NORD", 11: "CENTRE-SUD"
    }
    sector_options = {
        1: "Commerce", 2: "BTP", 3: "Industrie", 4: "Services", 5: "Agriculture",
        6: "Transport", 7: "Restauration", 8: "Immobilier", 9: "Finance", 10: "Santé",
        11: "Education", 12: "IT", 13: "Artisanat", 14: "Energie", 15: "Autres"
    }
    
    # Interface de saisie
    st.markdown("""
    <div class="input-group">
        <h3>📋 Saisie des Caractéristiques du Crédit</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💰 Variables Financières**")
        capital_paye = st.number_input("Capital Payé (t3)", min_value=0.0, value=50000.0, step=1000.0, format="%.4f")
        interet_paye = st.number_input("Intérêts Payés (t3)", min_value=0.0, value=5000.0, step=100.0, format="%.4f")
        montant_credit = st.number_input("Montant Crédit Total (t3)", min_value=0.0, value=100000.0, step=1000.0, format="%.4f")
        montant_debit = st.number_input("Montant Débit Total (t3)", min_value=0.0, value=80000.0, step=1000.0, format="%.4f")
        montant_moy = st.number_input("Montant Moyen (t3)", min_value=0.0, value=5000.0, step=100.0, format="%.4f")
        ecart_prev_reel = st.number_input("Écart Prévision/Réel (t3)", min_value=0.0, value=1000.0, step=100.0, format="%.4f")
        
    with col2:
        st.markdown("**📊 Ratios et Pourcentages**")
        ratio_interet = st.number_input("Ratio Intérêt/Capital (t3)", min_value=0.0, value=0.1, step=0.0001, format="%.4f")
        pct_interet = st.number_input("Pourcentage Intérêt Payé (t3)", min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")
        pct_capital = st.number_input("Pourcentage Capital Payé (t3)", min_value=0.0, max_value=1.0, value=0.6, step=0.0001, format="%.4f")
        retard_cumule_ratio = st.number_input("Ratio Retard Cumulé (t3)", min_value=0.0, value=0.05, step=0.0001, format="%.4f")
        ratio_debit_credit = st.number_input("Ratio Débit/Crédit (t3)", min_value=0.0, value=0.8, step=0.0001, format="%.4f")
        pct_retards = st.number_input("Pourcentage Retards (t3)", min_value=0.0, max_value=1.0, value=0.1, step=0.0001, format="%.4f")
        
    with col3:
        st.markdown("**🔢 Comportement de Paiement**")
        nb_ech_payees = st.number_input("Nb Échéances Payées (t3)", min_value=0, value=12, step=1)
        retard_moyen = st.number_input("Retard Moyen (jours t3)", min_value=0.0, value=5.0, step=0.1, format="%.1f")
        nb_mvts_credit = st.number_input("Nb Mouvements Crédit (t3)", min_value=0, value=10, step=1)
        nb_mvts_debit = st.number_input("Nb Mouvements Débit (t3)", min_value=0, value=8, step=1)
        duree_ecart = st.number_input("Durée Écart (t3)", min_value=0, value=30, step=1)
        
        st.markdown("**📅 Dates des Mouvements**")
        premier_mouvement_month = st.number_input("Mois Premier Mouvement (t3)", min_value=1, max_value=12, value=1)
        premier_mouvement_day = st.number_input("Jour Premier Mouvement (t3)", min_value=1, max_value=31, value=1)
        
        st.markdown("**👥 Informations Client**")
        client_type = st.selectbox("Type de Client", options=list(client_type_options.keys()), 
                                 format_func=lambda x: client_type_options[x])
        credit_type = st.selectbox("Type de Crédit", options=list(credit_type_options.keys()),
                                 format_func=lambda x: credit_type_options[x])
        region = st.selectbox("Région", options=list(region_options.keys()),
                            format_func=lambda x: region_options[x])
        sector = st.selectbox("Secteur d'Activité", options=list(sector_options.keys()),
                            format_func=lambda x: sector_options[x])
    
    # Modèle utilisé
    selected_model = 'XGBoost'
    st.info(f"**Modèle utilisé :** {selected_model}")
    
    enable_shap = st.checkbox("Activer l'analyse SHAP détaillée", value=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 Lancer la Prédiction et l'Analyse SHAP", 
                                 use_container_width=True, 
                                  type="primary",
                                  help="Cliquez pour analyser le risque de défaut avec explications détaillées")
    
    if predict_button:
        try:
            # ✅ VARIABLES EXACTES QUE LE MODÈLE ATTEND (24 colonnes seulement)
            expected_columns = [
                'capital_paye_t3', 'interet_paye_t3', 'nb_ech_payees_t3', 
                'nb_mvts_credit_t3', 'nb_mvts_debit_t3', 'pct_interet_paye_t3', 
                'retard_moyen_jours_t3', 'montant_moy_t3', 'montant_credit_total_t3', 
                'montant_debit_total_t3', 'pct_capital_paye_t3', 'ecart_prev_reel_t3', 
                'ratio_interet_sur_capital_paye_t3', 'retard_cumule_ratio_t3', 
                'ratio_debit_credit_t3', 'duree_ecart_t3', 'pct_retards_t3', 
                'premier_mouvement_t3_timestamp', 'premier_mouvement_t3_month', 
                'premier_mouvement_t3_day', 'client_type', 'credit_type', 
                'region', 'sector'
            ]
            
            # Calcul du timestamp seulement pour premier_mouvement
            import datetime
            premier_mouvement_date = datetime.datetime(2022, premier_mouvement_month, premier_mouvement_day)
            premier_timestamp = int(premier_mouvement_date.timestamp())
            
            # ✅ Préparation des données avec SEULEMENT les 24 colonnes nécessaires
            input_data = pd.DataFrame({
                'capital_paye_t3': [capital_paye],
                'interet_paye_t3': [interet_paye],
                'nb_ech_payees_t3': [nb_ech_payees],
                'nb_mvts_credit_t3': [nb_mvts_credit],
                'nb_mvts_debit_t3': [nb_mvts_debit],
                'pct_interet_paye_t3': [pct_interet],
                'retard_moyen_jours_t3': [retard_moyen],
                'montant_moy_t3': [montant_moy],
                'montant_credit_total_t3': [montant_credit],
                'montant_debit_total_t3': [montant_debit],
                'pct_capital_paye_t3': [pct_capital],
                'ecart_prev_reel_t3': [ecart_prev_reel],
                'ratio_interet_sur_capital_paye_t3': [ratio_interet],
                'retard_cumule_ratio_t3': [retard_cumule_ratio],
                'ratio_debit_credit_t3': [ratio_debit_credit],
                'duree_ecart_t3': [duree_ecart],
                'pct_retards_t3': [pct_retards],
                'premier_mouvement_t3_timestamp': [premier_timestamp],
                'premier_mouvement_t3_month': [premier_mouvement_month],
                'premier_mouvement_t3_day': [premier_mouvement_day],
                'client_type': [client_type],
                'credit_type': [credit_type],
                'region': [region],
                'sector': [sector]
            })
            
            # Réorganiser dans l'ordre exact
            input_data = input_data[expected_columns]
            
            # Vérification
            st.success(f"✅ Données préparées : {len(input_data.columns)} colonnes (exactement ce que le modèle attend)")
            
            # Prédiction
            model = models['XGBoost']
            
            # Extraction du modèle si c'est un dictionnaire
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'classifier' in model:
                    model = model['classifier']
                elif 'estimator' in model:
                    model = model['estimator']
                else:
                    model = list(model.values())[0]

            # Faire la prédiction
            prediction_encoded = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            prediction = le.inverse_transform([prediction_encoded])[0]
            
            # AFFICHAGE DES RÉSULTATS
            st.markdown("---")
            st.markdown('<div class="section-header">🎯 Résultat de la Prédiction</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 'SOLDE':
                    st.success(f"## ✅ CRÉDIT SOLDE")
                    st.metric("Probabilité de Solde", f"{proba[1]:.4%}", delta=f"Risque faible: {proba[0]:.4%}")
                else:
                    st.error(f"## 🚨 RISQUE DE DÉFAUT")
                    st.metric("Probabilité de Défaut", f"{proba[0]:.4%}", delta=f"Solde: {proba[1]:.4%}", delta_color="inverse")
                    
            with col2:
                risk_score = proba[0] * 100
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Score de Risque", 'font': {'size': 20}},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 50], 'color': "yellow"},
                            {'range': [50, 80], 'color': "orange"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True, key="risk_gauge")  # ← AJOUT key

            # ================================
            # ANALYSE SHAP (UNE SEULE SECTION)
            # ================================
            if enable_shap:
                try:
                    st.markdown("---")
                    st.markdown('<div class="section-header">🔍 Analyse Explicative SHAP</div>', unsafe_allow_html=True)
                    
                    with st.spinner("Calcul des explications SHAP..."):
                        explainer, shap_values = compute_shap_analysis(model, input_data, expected_columns, 'XGBoost')
                    
                    if shap_values is not None:
                        # Graphique summary
                        st.markdown("#### 📊 Impact des Variables sur la Décision")
                        shap_summary_fig = plot_shap_summary(shap_values, input_data, expected_columns)
                        if shap_summary_fig:
                            st.plotly_chart(shap_summary_fig, use_container_width=True, key="shap_summary")  # ← AJOUT key
                        
                        # Tableau détaillé
                        st.markdown("#### 📋 Détail des Contributions par Variable")
                        shap_table = create_shap_detailed_table(shap_values, input_data, expected_columns)
                        if shap_table is not None:
                            st.dataframe(shap_table.style.format({
                                'Valeur': '{:.4f}',
                                'Impact SHAP': '{:.6f}',
                                'Impact Absolu': '{:.6f}'
                            }).background_gradient(
                                subset=['Impact SHAP'], 
                                cmap='RdBu',
                                vmin=-0.1,
                                vmax=0.1
                            ), use_container_width=True, key="shap_table")  # ← AJOUT key
                        
                        # 3. Analyse des facteurs clés
                    st.markdown("#### 🎯 Facteurs Clés de la Décision")

                    if shap_table is not None:
                        # 🔧 CORRECTION : INVERSER l'affichage
                        # SHAP NÉGATIF = augmente le risque (pousse vers DÉFAUT)
                        # SHAP POSITIF = réduit le risque (pousse vers SOLDE)
                        top_risk_factors = shap_table[shap_table['Impact SHAP'] < 0].head(3)    # Facteurs de RISQUE
                        top_safety_factors = shap_table[shap_table['Impact SHAP'] > 0].head(3)  # Facteurs de SÉCURITÉ
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📈 Facteurs Augmentant le Risque de Défaut**")
                            if not top_risk_factors.empty:
                                for _, row in top_risk_factors.iterrows():
                                    st.write(f"• **{row['Variable']}**: {row['Impact SHAP']:.4f}")
                                    st.caption(f"Valeur: {row['Valeur']:.4f}")
                            else:
                                st.info("Aucun facteur n'augmente significativement le risque")
                        
                        with col2:
                            st.markdown("**📉 Facteurs Réduisant le Risque de Défaut**")
                            if not top_safety_factors.empty:
                                for _, row in top_safety_factors.iterrows():
                                    st.write(f"• **{row['Variable']}**: +{row['Impact SHAP']:.4f}")
                                    st.caption(f"Valeur: {row['Valeur']:.4f}")
                            else:
                                st.info("Aucun facteur ne réduit significativement le risque")
                        
                        # 4. Recommandations basées sur SHAP
                        st.markdown("#### 💡 Recommandations Stratégiques")
                        
                        if prediction == 'DEFAUT':
                            st.warning("""
                            **Actions recommandées pour réduire le risque:**
                            - Identifier et traiter les variables à fort impact positif sur le risque
                            - Mettre en place un suivi renforcé des indicateurs critiques
                            - Envisager des mesures correctives pour les ratios problématiques
                            """)
                        else:
                            st.success("""
                            **Points forts du dossier:**
                            - Les variables influencent positivement la solvabilité
                            - Le profil présente des caractéristiques favorables
                            - Possibilité d'envisager des conditions avantageuses
                            """)
                    
                    else:
                        st.info("L'analyse SHAP n'est pas disponible pour ce modèle ou cette configuration.")
                        
                except Exception as shap_error:
                    st.warning(f"⚠️ L'analyse SHAP n'a pas pu être générée: {shap_error}")
                    st.info("La prédiction a fonctionné, mais l'explication détaillée n'est pas disponible.")

            # Détails de la prédiction
            with st.expander("📊 Détails Complets de la Prédiction", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🤖 Informations du Modèle**")
                    st.info(f"**Modèle utilisé:** {selected_model}")
                    st.info(f"**Prédiction:** {prediction}")
                    
                    st.markdown("**📈 Probabilités Détaillées**")
                    prob_df = pd.DataFrame({
                        'Classe': le.classes_,
                        'Probabilité': proba
                    })
                    st.dataframe(prob_df.style.format({'Probabilité': '{:.4%}'}), use_container_width=True, key="prob_table")
                
                with col2:
                    st.markdown("**👤 Caractéristiques du Dossier**")
                    
                    characteristics = {
                        'Type de Client': client_type_options[client_type],
                        'Type de Crédit': credit_type_options[credit_type],
                        'Région': region_options[region],
                        'Secteur': sector_options[sector],
                        'Capital Payé': f"{capital_paye:,.4f}",
                        'Intérêts Payés': f"{interet_paye:,.4f}",
                        'Échéances Payées': f"{nb_ech_payees}",
                        'Retard Moyen': f"{retard_moyen} jours",
                        'Ratio Intérêt/Capital': f"{ratio_interet:.4f}",
                        'Pourcentage Capital Payé': f"{pct_capital:.4f}"
                    }
                    
                    for key, value in characteristics.items():
                        st.write(f"**{key}:** {value}")

            # Export de la prédiction avec données SHAP
            st.markdown("---")
            st.markdown("**💾 Export des Résultats**")
            
            prediction_data = {
                'timestamp': pd.Timestamp.now(),
                'model_utilise': selected_model,
                'prediction': prediction,
                'probabilite_defaut': f"{proba[0]:.6f}",
                'probabilite_solde': f"{proba[1]:.6f}",
                **input_data.iloc[0].to_dict()
            }
            
            # Ajouter les données SHAP si disponibles
            if enable_shap and 'shap_table' in locals() and shap_table is not None:
                for _, row in shap_table.iterrows():
                    prediction_data[f'shap_{row["Variable"]}'] = row['Impact SHAP']
            
            # Ajouter les libellés pour l'export
            prediction_data['client_type_label'] = client_type_options[client_type]
            prediction_data['credit_type_label'] = credit_type_options[credit_type]
            prediction_data['region_label'] = region_options[region]
            prediction_data['sector_label'] = sector_options[sector]
            
            prediction_df = pd.DataFrame([prediction_data])
            csv = prediction_df.to_csv(index=False, sep=';', decimal=',')
            
            st.download_button(
                label="📥 Télécharger le Rapport Complet (CSV)",
                data=csv,
                file_name=f"prediction_credit_shap_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_button"
            )

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
       


# ================================
# FONCTION PRINCIPALE
# ================================

def main():
    """Fonction principale de l'application"""
    
    # Initialisation de la session state
    if 'page' not in st.session_state:
        st.session_state.page = 'Accueil'
    
    # Navigation verticale dans la sidebar
    show_sidebar_navigation()
    
    # Affichage de la page sélectionnée
    if st.session_state.page == 'Accueil':
        show_home()
    elif st.session_state.page == 'Analyse Exploratoire':
        show_exploratory_analysis()
    elif st.session_state.page == 'Statistiques Descriptives':
        show_descriptive_stats()
    elif st.session_state.page == 'Performance des Modèles':
        show_model_performance()
    elif st.session_state.page == 'Prédictions':
        show_predictions()
    
    # Pied de page
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 2rem;'>"
        "🏦 <strong>Application d'Analyse de Risque de Crédit avec SHAP</strong> • "
        f"Dernière mise à jour: {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )

# ================================
# LANCEMENT DE L'APPLICATION
# ================================

if __name__ == "__main__":
    main()
