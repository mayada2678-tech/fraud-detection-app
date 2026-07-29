import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# =========================================================
# KONFIGURATION & MODELL-PARAMETER
# =========================================================
st.set_page_config(page_title="Fraud Detector - CSV & Verhalten", layout="wide")

# 🎯 Optimaler Schwellenwert aus der F1-Score-Optimierung
MSE_THRESHOLD = 9.25

# =========================================================
# HELFERFUNKTION: MODELL & SCALER LADEN
# =========================================================
@st.cache_resource
def load_ml_pipeline():
    """
    Lädt das Autoencoder-Modell und den PowerTransformer/Scaler.
    Ersetze die Dateipfade bei Bedarf durch deine echten Pfade.
    """
    try:
        # Autoencoder Keras Modell
        autoencoder = tf.keras.models.load_model("autoencoder_model.h5")
        
        # PowerTransformer / Scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        return autoencoder, scaler
    except Exception as e:
        # Falls Dateien nicht gefunden wurden, geben wir None zurück (für Demo-Betrieb)
        return None, None

autoencoder, scaler = load_ml_pipeline()

# =========================================================
# HEADER & TITEL
# =========================================================
st.title("💳 Betrugsdetektor: CSV-Historie & Autoencoder-Prüfung")
st.write("Dieses System kombiniert **Autoencoder-Anomalieerkennung** mit **regelbasierter CSV-Verhaltensanalyse**.")
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
    data = {
        "transaktion_id": [f"TX{i}" for i in range(1, 21)],
        "betrag": [15.50, 22.00, 8.90, 45.00, 12.00, 120.00, 18.50, 25.00, 30.00, 14.20,
                   19.99, 85.00, 11.00, 40.00, 15.00, 22.50, 50.00, 9.90, 35.00, 28.00],
        "kategorie": ["Supermarkt", "Tankstelle", "Bäcker", "Kleidung", "Bäcker", "Elektronik", 
                      "Supermarkt", "Restaurant", "Supermarkt", "Bäcker", "Online", "Kleidung", 
                      "Supermarkt", "Restaurant", "Bäcker", "Tankstelle", "Online", "Supermarkt", "Kleidung", "Supermarkt"]
    }
    df_history = pd.DataFrame(data)

with st.expander("📊 Kundendaten & Historie aus der CSV anzeigen"):
    st.dataframe(df_history)

# =========================================================
# AUTOMATISCHE PROFIL-BERECHNUNG AUS DER CSV
# =========================================================
user_avg_amount = df_history["betrag"].mean()
user_max_amount = df_history["betrag"].max()
user_total_spent = df_history["betrag"].sum()
total_transactions = len(df_history)

estimated_balance = max(2000.0, user_total_spent * 1.5)

st.write("---")
st.header("2. Berechnetes Kundenprofil (aus CSV ermittelt)")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("Ø Ausgaben / Kauf", f"{user_avg_amount:.2f} €")
col_p2.metric("Höchster Bisheriger Kauf", f"{user_max_amount:.2f} €")
col_p3.metric("Gesamtzahl Käufe (CSV)", f"{total_transactions}")
col_p4.metric("Geschätztes Rahmen/Guthaben", f"{estimated_balance:.2f} €")

# =========================================================
# SCHRITT 2: Neue Transaktion eingeben
# =========================================================
st.write("---")
st.header("3. Neue Transaktion zur Beurteilung eingeben")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    new_tx_amount = st.number_input("Neuer Kaufbetrag (€)", min_value=1.0, max_value=50000.0, value=1000.0, step=10.0)

with col_in2:
    new_tx_location = st.selectbox("Standort", ["Inland (Normal)", "Ausland (Online)", "Risikoland"])

with col_in3:
    new_tx_24h_count = st.slider("Weitere Käufe in den letzten 24h", 0, 20, 1)

# =========================================================
# SCHRITT 3: LOGIK & BEURTEILUNG (Autoencoder MSE & Regelsystem)
# =========================================================
st.write("---")
st.header("⚖️ Beurteilung der neuen Transaktion")

ratio = new_tx_amount / user_avg_amount if user_avg_amount > 0 else 1.0
risk_score = 0.0
reasons = []

# ---------------------------------------------------------
# 🤖 AUTOENCODER PRÜFUNG (MSE vs Threshold = 9.25)
# ---------------------------------------------------------
mse_calculated = None

if autoencoder is not None and scaler is not None:
    # 1. Beispieldaten für Transaktions-Features erstellen (29/30 Features wie im CreditCard Dataset)
    # Ersetze dies durch deine tatsächlichen Input-Features
    dummy_input = np.zeros((1, scaler.n_features_in_)) 
    dummy_input[0, -1] = new_tx_amount  # Setze Betrag an die Stelle des Betrags-Features
    
    # 2. Skalierung
    scaled_input = scaler.transform(dummy_input)
    
    # 3. Vorhersage / Rekonstruktion
    reconstruction = autoencoder.predict(scaled_input, verbose=0)
    
    # 4. Berechnung des Rekonstruktionsfehlers (MSE)
    mse_calculated = np.mean(np.power(scaled_input - reconstruction, 2), axis=1)[0]
    
    # 5. Bewertung basierend auf MSE_THRESHOLD = 9.25
    if mse_calculated >= MSE_THRESHOLD:
        risk_score += 0.60
        reasons.append(f"🚨 **Autoencoder-Anomalie:** Rekonstruktionsfehler (MSE: **{mse_calculated:.2f}**) überschreitet den Schwellenwert von **{MSE_THRESHOLD}**!")
    else:
        reasons.append(f"🟢 **Autoencoder Unauffällig:** Rekonstruktionsfehler (MSE: **{mse_calculated:.2f}**) liegt unter dem Schwellenwert von **{MSE_THRESHOLD}**.")
else:
    # Simulation, falls kein .h5 / .pkl File geladen ist (Dummy-MSE für Live-Tests)
    # Erzeugt höheren MSE bei extrem abweichenden Beträgen
    simulated_mse = float((new_tx_amount / (user_avg_amount + 1e-5)) * 0.5)
    mse_calculated = simulated_mse
    
    if simulated_mse >= MSE_THRESHOLD:
        risk_score += 0.60
        reasons.append(f"🚨 **Autoencoder-Anomalie (Simuliert):** MSE **{simulated_mse:.2f}** >= Threshold **{MSE_THRESHOLD}**!")
    else:
        reasons.append(f"🟢 **Autoencoder Unauffällig (Simuliert):** MSE **{simulated_mse:.2f}** < Threshold **{MSE_THRESHOLD}**.")

# ---------------------------------------------------------
# 📊 BRANCHEN- & VERHALTENSREGELN (CSV-Vergleich)
# ---------------------------------------------------------
# Regel 1: Betrag weicht extrem vom CSV-Durchschnitt ab
if ratio >= 20:
    risk_score += 0.50
    reasons.append(f"🚨 **Extreme Abweichung:** Der Betrag ({new_tx_amount:.2f} €) ist **{ratio:.1f}-mal höher** als der CSV-Durchschnitt ({user_avg_amount:.2f} €)!")
elif ratio >= 5:
    risk_score += 0.25
    reasons.append(f"⚠️ **Erhöhte Abweichung:** Der Betrag ist {ratio:.1f}-mal höher als der normale Kundendurchschnitt ({user_avg_amount:.2f} €).")

# Regel 2: Betrag übersteigt den höchsten jemals getätigten Kauf aus der CSV deutlich
if new_tx_amount > (user_max_amount * 3):
    risk_score += 0.30
    reasons.append(f"🚨 **Rekordkauf:** Dieser Kauf übertrifft den höchsten bisherigen CSV-Kauf ({user_max_amount:.2f} €) um mehr als das 3-fache.")

# Regel 3: Guthaben / Verfügungsrahmen im Vergleich zum Betrag
if new_tx_amount > estimated_balance:
    risk_score += 0.40
    reasons.append(f"🚨 **Guthaben überschritten:** Der Kaufbetrag liegt über dem geschätzten Budget ({estimated_balance:.2f} €).")
elif estimated_balance >= 5000 and new_tx_amount <= 1000:
    risk_score = max(0.0, risk_score - 0.20)
    reasons.append("🟢 **Hohes Guthaben:** Kunde hat ausreichend Rahmen für Käufe bis 1.000 €.")

# Regel 4: Standort
if new_tx_location == "Risikoland":
    risk_score += 0.35
    reasons.append("⚠️ **Risikoland:** Transaktion kommt aus einer verdächtigen Region.")

# Prozentualer Risiko-Score
fraud_probability = min(1.0, risk_score) * 100

# =========================================================
# ERGEBNIS-AUSGABE
# =========================================================
res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.metric("Berechnetes Gesamtrisiko", f"{fraud_probability:.1f} %")
    if mse_calculated is not None:
        st.metric("Berechneter MSE (Autoencoder)", f"{mse_calculated:.2f}", delta=f"Threshold: {MSE_THRESHOLD}", delta_color="inverse")
    
    if fraud_probability < 35:
        st.success("✅ **STATUS: FREIGEGEBEN (APPROVE)**")
        st.caption("Verhalten entspricht dem Kundenprofil aus der CSV.")
    elif fraud_probability < 70:
        st.warning("⚠️ **STATUS: PRÜFUNG ERFORDERLICH (2FA / TAN)**")
        st.caption("Sicherheitsprüfung notwendig (SMS-TAN senden).")
    else:
        st.error("🚨 **STATUS: BLOCKIERT (BLOCK)**")
        st.caption("Transaktion wird wegen hoher Abweichung gestoppt.")

with res_col2:
    st.subheader("Begründung & Detail-Analyse:")
    for r in reasons:
        st.write(r)
