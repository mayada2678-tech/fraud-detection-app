import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# =========================================================
# SEITEN-KONFIGURATION
# =========================================================
st.set_page_config(page_title="Fraud Detector - CSV & Verhalten", layout="wide")

# 🎯 Optimaler Schwellenwert aus deiner Schwellenwert-Analyse / F1-Kurve
MSE_THRESHOLD = 9.25

# =========================================================
# HELFERFUNKTION: ML-MODELL & SCALER LADEN
# =========================================================
@st.cache_resource
def load_ml_pipeline():
    """
    Lädt das trainierte Autoencoder-Modell und den Scaler (pickle).
    """
    autoencoder = None
    scaler = None
    
    if os.path.exists("autoencoder_model.h5"):
        try:
            autoencoder = tf.keras.models.load_model("autoencoder_model.h5")
        except Exception:
            pass
            
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            pass
            
    return autoencoder, scaler

autoencoder, scaler = load_ml_pipeline()

# =========================================================
# HEADER & TITEL (wie im Screenshot)
# =========================================================
st.title("💳 Betrugsdetektor: CSV-Historie & Live-Prüfung")
st.write("Dieses System analysiert die Historie eines Kunden aus einer CSV-Datei und beurteilt eine **neue Transaktion** anhand seines bisherigen Verhaltens.")
st.write("---")

# =========================================================
# SCHRITT 1: CSV Hochladen (Kunden-Historie)
# =========================================================
st.header("1. Kunden-Historie laden (CSV)")

uploaded_file = st.file_uploader("Lade die Transaktions-Historie des Kunden hoch (.csv)", type=["csv"])

if uploaded_file is not None:
    df_history = pd.read_csv(uploaded_file)
    st.success("✅ CSV-Historie erfolgreich geladen!")
else:
    st.info("ℹ️ Keine CSV hochgeladen. Es werden Demo-Historien-Daten genutzt.")
    # Erstelle Beispiel-Historie eines normalen Kunden
    data = {
        "transaktion_id": [f"TX{i}" for i in range(1, 21)],
        "betrag": [15.50, 22.00, 8.90, 45.00, 12.00, 120.00, 18.50, 25.00, 30.00, 14.20,
                   19.99, 85.00, 11.00, 40.00, 15.00, 22.50, 50.00, 9.90, 35.00, 28.00],
        "kategorie": ["Supermarkt", "Tankstelle", "Bäcker", "Kleidung", "Bäcker", "Elektronik", 
                      "Supermarkt", "Restaurant", "Supermarkt", "Bäcker", "Online", "Kleidung", 
                      "Supermarkt", "Restaurant", "Bäcker", "Tankstelle", "Online", "Supermarkt", "Kleidung", "Supermarkt"]
    }
    df_history = pd.DataFrame(data)

# Historie in einem ausklappbaren Bereich anzeigen
with st.expander("📊 Kundendaten & Historie aus der CSV anzeigen"):
    st.dataframe(df_history)

# =========================================================
# AUTOMATISCHE PROFIL-BERECHNUNG AUS DER CSV
# =========================================================
# Flexible Erkennung der Betrags-Spalte (für creditcard_small.csv / Amount vs. betrag)
amount_col = None
for col in ["Amount", "betrag", "Amount (€)", "Betrag", "amount"]:
    if col in df_history.columns:
        amount_col = col
        break

if amount_col is not None:
    user_avg_amount = float(df_history[amount_col].mean())
    user_max_amount = float(df_history[amount_col].max())
    user_total_spent = float(df_history[amount_col].sum())
    total_transactions = len(df_history)
else:
