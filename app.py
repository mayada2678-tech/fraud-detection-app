# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 07:12:26 2026

@author: mayad
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import os 

# ==============================================================================
# 1. SEITEN-KONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS für ansprechendes Design
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .status-green { color: #28a745; font-weight: bold; }
    .status-yellow { color: #ffc107; font-weight: bold; }
    .status-red { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODELL & PIPELINE LADEN (MIT CACHING & ABSOLUTEN PFADEN)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_pt.pkl")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "calibrator.pkl")

@st.cache_resource
def load_artifacts():
    # Autoencoder
    model = tf.keras.models.load_model(MODEL_PATH) 
    
    # Scaler (PowerTransformer)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    # Calibrator (Platt Scaling)
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)
        
    return model, scaler, calibrator

try:
    autoencoder, scaler, calibrator = load_artifacts()
    st.sidebar.success("✅ KI-Modelle & Skalierer geladen")
except Exception as e:
    st.error(f"❌ Fehler beim Laden der Modelldateien: {e}")
    st.info("Bitte stelle sicher, dass 'autoencoder_model.keras', 'scaler_pt.pkl' und 'calibrator.pkl' im selben Ordner liegen.")
    st.stop()

# ==============================================================================
# 3. HEADER & SEITENLEISTE (SZENARIEN)
# ==============================================================================
st.title("🛡️ KI-System zur Echtzeit-Betrugserkennung")
st.caption("Autoencoder Anomaly Detection mit kalibrierten Betrugswahrscheinlichkeiten (Platt Scaling)")

st.sidebar.header("⚙️ Operative Strategie")
scenario = st.sidebar.radio(
    "Wähle den Risiko-Modus:",
    ["Standard (Ausgewogen)", "Aggressiv (Fokus auf Workload-Minimierung)", "Sicher (Maximale Betrugserkennung)", "Custom (Manuell)"]
)

# Schwellenwerte basierend auf der Analyse setzen
if scenario == "Sicher (Maximale Betrugserkennung)":
    yellow_thresh = 0.10
    red_thresh = 0.20
    st.sidebar.info("💡 **Recall: ~90%** | Fokus auf maximalen Schutz.")
elif scenario == "Standard (Ausgewogen)":
    yellow_thresh = 0.20
    red_thresh = 0.50
    st.sidebar.info("💡 **Recall: ~84%** | Ausgewogenes Verhältnis aus Schutz & Aufwand.")
elif scenario == "Aggressiv (Fokus auf Workload-Minimierung)":
    yellow_thresh = 0.50
    red_thresh = 0.80
    st.sidebar.info("💡 **Recall: ~82%** | Reduziert Fehlalarme um >60%!")
else:
    yellow_thresh = st.sidebar.slider("Warnschwelle (2FA)", 0.05, 0.50, 0.20, step=0.05)
    red_thresh = st.sidebar.slider("Sperrschwelle (Block)", 0.30, 0.95, 0.80, step=0.05)

# ==============================================================================
# 4. TAB-NAVIGATION
# ==============================================================================
tab1, tab2 = st.tabs(["🔍 Einzeltransaktion Prüfen", "📊 Batch-Analyse & Strategie-Vergleich"])

# ------------------------------------------------------------------------------
# TAB 1: EINZELPRÜFUNG (LIVE-DEMO)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Transaktionsdaten eingeben")
    
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaktionsbetrag (€)", min_value=0.0, value=149.99, step=10.0)
    with col2:
        v1_sim = st.slider("Auffälligkeits-Indikator V1 (Simuliert)", -5.0, 5.0, 0.0)

    # Zufällige Feature-Vektoren für V1..V28 (29 Features insgesamt inkl. Amount)
    # Für die Demo: Wir bauen eine Test-Zeile auf
    if st.button("🚀 Transaktion analysieren", type="primary"):
        # Erstelle Beispiel-Feature-Array (29 Features: V1-V28 + Amount)
        # Normalverteilte Dummy-Werte um 0
        sample_features = np.zeros((1, 29))
        sample_features[0, 0] = v1_sim  # V1
        sample_features[0, -1] = amount # Amount
        
        # 1. Skalieren mit PowerTransformer
        scaled_features = scaler.transform(sample_features)
        
        # 2. Rekonstruktionsfehler (MSE) berechnen
        pred = autoencoder.predict(scaled_features)
        mse = np.mean(np.square(scaled_features - pred), axis=1)
        
        # 3. Wahrscheinlichkeit via Kalibrator berechnen
        prob = calibrator.predict_proba(mse.reshape(-1, 1))[0, 1]
        
        # AUSWERTUNG & VISUALISIERUNG
        st.markdown("---")
        st.subheader("Ergebnis der KI-Prüfung")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Rekonstruktionsfehler (MSE)", f"{mse[0]:.4f}")
        m_col2.metric("Betrugswahrscheinlichkeit", f"{prob * 100:.2f} %")
        
        # Decision Logic (3-Stufen Ampel)
        if prob >= red_thresh:
            m_col3.error("🔴 ACTION: BLOCK (Betrugsverdacht)")
            st.error(f"⚠️ **HOCHES RISIKO ({prob*100:.1f}%):** Die Transaktion überschreitet die Sperrschwelle von {red_thresh*100:.0f}%. Sie wurde blockiert und an das Fraud-Team weitergeleitet.")
        elif prob >= yellow_thresh:
            m_col3.warning("🟡 ACTION: 2FA PRÜFUNG")
            st.warning(f"⚡ **MITTLERES RISIKO ({prob*100:.1f}%):** Die Transaktion liegt über der Warnschwelle von {yellow_thresh*100:.0f}%. Der Kunde muss sich per 2FA/SMS verifizieren.")
        else:
            m_col3.success("🟢 ACTION: FREIGABE")
            st.success(f"✅ **GERINGES RISIKO ({prob*100:.1f}%):** Transaktion unauffällig. Automatisch freigegeben.")

        # Tacho-Chart / Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Betrugs-Risiko (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps' : [
                    {'range': [0, yellow_thresh*100], 'color': "#28a745"},
                    {'range': [yellow_thresh*100, red_thresh*100], 'color': "#ffc107"},
                    {'range': [red_thresh*100, 100], 'color': "#dc3545"}
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: BATCH-ANALYSE & BUSINESS-IMPACT
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📁 CSV-Datei für Batch-Verarbeitung hochladen")
    uploaded_file = st.file_content = st.file_uploader("Upload 'creditcard.csv' oder Test-Subset", type=["csv"])
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        
        if "Time" in df_batch.columns:
            df_batch = df_batch.drop(columns=["Time"])
            
        X_batch = df_batch.drop(columns=["Class"], errors='ignore').values
        
        with st.spinner("Analysiere gesamte Datei..."):
            # Transformation & Prediction
            X_batch_scaled = scaler.transform(X_batch)
            pred_batch = autoencoder.predict(X_batch_scaled)
            mse_batch = np.mean(np.square(X_batch_scaled - pred_batch), axis=1)
            probs_batch = calibrator.predict_proba(mse_batch.reshape(-1, 1))[:, 1]
            
            df_batch["Fraud_Probability"] = probs_batch
            df_batch["Status"] = np.where(probs_batch >= red_thresh, "🔴 Blockieren", 
                                 np.where(probs_batch >= yellow_thresh, "🟡 2FA Anfordern", "🟢 Freigeben"))
        
        st.success(f"Erfolgreich {len(df_batch):,} Transaktionen analysiert!")
        
        # KPI Kacheln
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        green_cnt = (df_batch["Status"] == "🟢 Freigeben").sum()
        yellow_cnt = (df_batch["Status"] == "🟡 2FA Anfordern").sum()
        red_cnt = (df_batch["Status"] == "🔴 Blockieren").sum()
        
        b_col1.metric("Gesamt", f"{len(df_batch):,}")
        b_col2.metric("🟢 Automatisch Freigegeben", f"{green_cnt:,}", f"{green_cnt/len(df_batch)*100:.1f}%")
        b_col3.metric("🟡 2FA Prüfungen", f"{yellow_cnt:,}", f"{yellow_cnt/len(df_batch)*100:.1f}%")
        b_col4.metric("🔴 Manuelle Reviews / Block", f"{red_cnt:,}", f"{red_cnt/len(df_batch)*100:.1f}%")
        
        # Verteilung visualisieren
        fig_pie = px.pie(
            df_batch, 
            names="Status", 
            title="Verteilung der operativen Aktionen",
            color="Status",
            color_discrete_map={"🟢 Freigeben": "#28a745", "🟡 2FA Anfordern": "#ffc107", "🔴 Blockieren": "#dc3545"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Vorschau der kritischen Fälle
        st.subheader("🔴 Liste der blockierten Verdachtsfälle (Top Risiko)")
        st.dataframe(df_batch[df_batch["Status"] == "🔴 Blockieren"].sort_values(by="Fraud_Probability", ascending=False).head(10))
