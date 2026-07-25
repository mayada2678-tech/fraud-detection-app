import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Echtzeit Fraud Detector", layout="wide")

st.title("💳 Echtzeit-Kreditkarten-Betrugsdetektor")
st.subheader("Verhaltensbasierte Transaktionsprüfung (Behavioral Fraud Engine)")
st.write("---")

# ---------------------------------------------------------
# EINGABEMASKE (Kundenprofil & Transaktionsdaten)
# ---------------------------------------------------------
st.header("🔍 Neue Transaktion prüfen")

col1, col2, col3 = st.split_columns(3) if hasattr(st, 'split_columns') else st.columns(3)

with col1:
    st.subheader("1. Die aktuelle Transaktion")
    tx_amount = st.number_input("Aktueller Betrag (€)", min_value=1.0, max_value=50000.0, value=1000.0, step=10.0)
    tx_location = st.selectbox("Standort der Zahlung", ["Inland (Normal)", "Ausland (Online)", "Risikoland (Sperrgebiet)"])
    tx_count_24h = st.slider("Transaktionen in den letzten 24 Std.", 1, 30, 2)

with col2:
    st.subheader("2. Kunden-Historie & Profil")
    user_avg_amount = st.number_input("Ø Betrag des Kunden bisher (€)", min_value=1.0, max_value=10000.0, value=25.0, step=5.0)
    user_balance = st.number_input("Geschätztes Guthaben / Rahmen (€)", min_value=0.0, max_value=100000.0, value=1500.0, step=500.0)
    user_risk_class = st.selectbox("Kunden-Risikoklasse", ["Standard", "VIP / Premium", "Neukunde (Unbekannt)"])

with col3:
    st.subheader("3. Muster-Anomalie (KI-Merkmal)")
    # V1 bis V28 Abweichung vereinfacht als ein Schieberegler simulieren
    pattern_anomaly = st.slider("KI-Anomaliewert (Musterabweichung)", 0.0, 10.0, 1.2, help="Hohe Werte bedeuten unnormale technische Merkmale (z.B. IP, Device, Verschlüsselung).")

st.write("---")

# ---------------------------------------------------------
# BEWERTUNGSLOGIK (Echte Regeln + KI-Score)
# ---------------------------------------------------------

# Berechnete Faktoren
ratio = tx_amount / user_avg_amount if user_avg_amount > 0 else 999.0

# Risikopunkte sammeln
risk_score = 0.0
reasons = []

# Regel 1: Abweichung vom normalen Kaufverhalten
if ratio > 10 and user_balance < 3000:
    risk_score += 0.45
    reasons.append(f"⚠️ Der Betrag ist **{ratio:.1f}-mal höher** als der gewohnte Schnitt ({user_avg_amount} €) bei geringem Guthaben.")

# Regel 2: Betrag übersteigt Guthaben/Rahmen deutlich
if tx_amount > user_balance:
    risk_score += 0.35
    reasons.append("⚠️ Transaktionsbetrag übersteigt das verfügbare Kunden-Guthaben/Limit!")

# Regel 3: Ungewöhnliche Frequenz
if tx_count_24h > 10:
    risk_score += 0.25
    reasons.append(f"⚠️ Ungewöhnlich viele Transaktionen ({tx_count_24h} Käufe) in 24 Stunden.")

# Regel 4: Standort
if tx_location == "Risikoland (Sperrgebiet)":
    risk_score += 0.40
    reasons.append("⚠️ Zahlung stammt aus einem Risikoland.")

# Regel 5: Technische Muster-Anomalie (PCA / KI)
if pattern_anomaly > 4.0:
    risk_score += 0.30
    reasons.append("⚠️ Technische KI-Analyse erkennt verdächtige Transaktions-Header/IP-Muster.")

# VIP Bonus (Toleranter bei reichen Kunden)
if user_risk_class == "VIP / Premium" and tx_amount <= user_balance:
    risk_score = max(0.0, risk_score - 0.20)

# Begrenzung auf 0 % bis 100 %
fraud_probability = min(1.0, risk_score) * 100

# ---------------------------------------------------------
# ERGEBNIS-ANZEIGE
# ---------------------------------------------------------
st.header("📊 Analyse-Ergebnis")

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.metric("Berechnetes Risiko", f"{fraud_probability:.1f} %")
    
    if fraud_probability < 30:
        st.success("✅ **STATUS: FREIGEGEBEN (APPROVE)**")
        st.caption("Transaktion wird ohne Störung verarbeitet.")
    elif fraud_probability < 70:
        st.warning("⚠️ **STATUS: PRÜFUNG ERFORDERLICH (2FA / TAN)**")
        st.caption("Kunde muss die Zahlung in der Banking-App mit SMS/Push-TAN bestätigen.")
    else:
        st.error("🚨 **STATUS: BLOCKIERT (BLOCK)**")
        st.caption("Transaktion wurde wegen Betrugsverdacht gestoppt!")

with res_col2:
    st.subheader("Begründung & Risiko-Faktoren:")
    if len(reasons) == 0:
        st.write("🟢 Keine Auffälligkeiten. Das Verhalten entspricht genau dem üblichen Kundenprofil.")
    else:
        for r in reasons:
            st.write(r)

# ---------------------------------------------------------
# SIMULATIONEN ZUM SCHNELLTESTEN
# ---------------------------------------------------------
st.write("---")
st.subheader("💡 Schnelltest-Szenarien")
st.caption("Vergleiche das System mit deinen zwei Gedanken-Beispielen:")

sc1, sc2 = st.columns(2)

with sc1:
    st.info("**Szenario A (Kunde mit wenig Guthaben):**\n- Kauf: 1.000 €\n- Ø Einkauf: 25 €\n- Guthaben: 1.500 €\n👉 **Ergebnis:** Das System schlägt Alarm (Verhaltensabweichung).")

with sc2:
    st.success("**Szenario B (Kunde mit viel Guthaben / VIP):**\n- Kauf: 1.000 €\n- Ø Einkauf: 300 €\n- Guthaben: 8.000 €\n👉 **Ergebnis:** Das System gibt die Transaktion problemlos frei.")
