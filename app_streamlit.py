"""
app_streamlit.py — Interfaccia web opzionale (Streamlit).

Avvio:
    streamlit run app_streamlit.py

Usa gli stessi moduli della CLI: cambia solo la presentazione.
Qui l'input è sempre manuale (più pratico per uso interattivo);
per l'integrazione API usa la CLI o estendi questo file.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_fetcher import QuoteMatch, StatisticheSquadra
from src.models import monte_carlo, poisson_model

st.set_page_config(page_title="Simulatore Calcio", page_icon="⚽")
st.title("⚽ Simulatore Predittivo di Partite")
st.caption("Poisson calibrato sulle quote + Monte Carlo Gamma-Poisson (10.000 iterazioni)")

# --------------------------- INPUT MATCH -----------------------------------
col_a, col_b = st.columns(2)
with col_a:
    competizione = st.text_input("Competizione", "Serie A")
    casa = st.text_input("Squadra di casa", "Roma")
with col_b:
    data_match = st.date_input("Data del match")
    fuori = st.text_input("Squadra fuori casa", "Lazio")

st.subheader("Statistiche squadre")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**{casa} (in casa)**")
    gf_c = st.number_input("Gol fatti / partita (casa)", 0.0, 5.0, 1.6, 0.1)
    gs_c = st.number_input("Gol subiti / partita (casa)", 0.0, 5.0, 1.0, 0.1)
    forma_c = st.text_input("Forma ultime 5 (es. WWDLW)", "WWDLW", key="fc")
with c2:
    st.markdown(f"**{fuori} (fuori casa)**")
    gf_f = st.number_input("Gol fatti / partita (fuori)", 0.0, 5.0, 1.2, 0.1)
    gs_f = st.number_input("Gol subiti / partita (fuori)", 0.0, 5.0, 1.4, 0.1)
    forma_f = st.text_input("Forma ultime 5 (es. LWDWL)", "LWDWL", key="ff")

st.subheader("Quote bookmaker (opzionali)")
usa_quote = st.checkbox("Calibra il modello sulle quote di mercato", value=True)
quote = None
if usa_quote:
    q1, q2, q3 = st.columns(3)
    quota_1 = q1.number_input("Quota 1", 1.01, 50.0, 2.30, 0.05)
    quota_x = q2.number_input("Quota X", 1.01, 50.0, 3.30, 0.05)
    quota_2 = q3.number_input("Quota 2", 1.01, 50.0, 3.10, 0.05)
    q4, q5 = st.columns(2)
    quota_o = q4.number_input("Quota Over 2.5", 1.01, 10.0, 1.90, 0.05)
    quota_u = q5.number_input("Quota Under 2.5", 1.01, 10.0, 1.90, 0.05)
    quote = QuoteMatch(quota_1, quota_x, quota_2, quota_o, quota_u)

# --------------------------- SIMULAZIONE -----------------------------------
if st.button("🎲 Simula la partita", type="primary"):
    st_casa = StatisticheSquadra(casa, gf_c, gs_c, list(forma_c.upper()))
    st_fuori = StatisticheSquadra(fuori, gf_f, gs_f, list(forma_f.upper()))

    lam_c, lam_f = poisson_model.lambdas_da_statistiche(st_casa, st_fuori)
    lam_c, lam_f = poisson_model.calibra_con_quote(lam_c, lam_f, quote)
    lam_c, lam_f = poisson_model.applica_forma(lam_c, lam_f, st_casa, st_fuori)

    ris = monte_carlo.esegui_monte_carlo(lam_c, lam_f)

    st.metric("Gol attesi", f"{casa} {lam_c:.2f} — {lam_f:.2f} {fuori}")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"1 · {casa}", f"{ris.p_1:.1%}")
    m2.metric("X · Pareggio", f"{ris.p_x:.1%}")
    m3.metric(f"2 · {fuori}", f"{ris.p_2:.1%}")

    st.subheader("Top 3 risultati esatti")
    top = ris.top_risultati_esatti(3)
    st.table(pd.DataFrame(
        [(s, f"{p:.1%}") for s, p in top],
        columns=["Risultato", "Probabilità"],
    ))

    m4, m5, m6, m7 = st.columns(4)
    m4.metric("Over 2.5", f"{ris.p_over25:.1%}")
    m5.metric("Under 2.5", f"{ris.p_under25:.1%}")
    m6.metric("Goal (GG)", f"{ris.p_goal:.1%}")
    m7.metric("NoGoal (NG)", f"{ris.p_nogoal:.1%}")

    st.subheader("Distribuzione gol totali")
    dist = ris.distribuzione_gol_totali()
    st.bar_chart(pd.DataFrame(
        {"Probabilità": list(dist.values())},
        index=[f"{g}+" if g == max(dist) else str(g) for g in dist],
    ))

    st.info("Probabilità descrittive del mercato, non consigli di scommessa. "
            "Un modello calibrato sulle quote non può batterle.")
