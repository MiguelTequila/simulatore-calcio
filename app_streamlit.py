"""
app_streamlit.py — Simulatore Predittivo di Partite (versione a file unico).

Autosufficiente: modelli Poisson + Monte Carlo + interfaccia in un solo
file, senza dipendenze dal pacchetto src/. Pensato per girare sia con
`streamlit run` sia nel browser via stlite su GitHub Pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import poisson, skellam

# ===========================================================================
# PARAMETRI DEL MODELLO
# ===========================================================================
N_SIMULAZIONI = 10_000      # iterazioni Monte Carlo
PESO_QUOTE = 0.70           # peso del mercato vs modello statistico (0..1)
IMPATTO_FORMA_MAX = 0.10    # la forma sposta i gol attesi al massimo del ±10%
GAMMA_DISPERSIONE_K = 15.0  # incertezza sui λ (più basso = code più grasse)
MEDIA_GOL_CASA_LEGA = 1.50  # medie di lega usate come riferimento
MEDIA_GOL_FUORI_LEGA = 1.20


# ===========================================================================
# STRUTTURE DATI
# ===========================================================================

@dataclass
class StatisticheSquadra:
    """Statistiche essenziali di una squadra per il modello."""
    nome: str
    gol_fatti_media: float
    gol_subiti_media: float
    forma_ultime5: list = field(default_factory=list)

    def punteggio_forma(self) -> float:
        """Forma -> moltiplicatore centrato su 1.0 (W=3, D=1, L=0)."""
        if not self.forma_ultime5:
            return 1.0
        punti = {"W": 3, "D": 1, "L": 0}
        totale = sum(punti.get(r.upper(), 1) for r in self.forma_ultime5)
        massimo = 3 * len(self.forma_ultime5)
        atteso = 1.4 * len(self.forma_ultime5)
        scost = (totale - atteso) / (massimo - atteso)
        scost = max(-1.0, min(1.0, scost))
        return 1.0 + scost * IMPATTO_FORMA_MAX


@dataclass
class QuoteMatch:
    """Quote decimali del bookmaker per il match."""
    quota_1: float
    quota_x: float
    quota_2: float
    quota_over25: float | None = None
    quota_under25: float | None = None

    def probabilita_implicite_1x2(self):
        """Quote 1X2 -> probabilità de-marginate (metodo proporzionale)."""
        raw = [1 / self.quota_1, 1 / self.quota_x, 1 / self.quota_2]
        margine = sum(raw)
        return tuple(p / margine for p in raw)

    def probabilita_implicita_over25(self):
        """Probabilità de-marginata dell'Over 2.5 (None se quote assenti)."""
        if not self.quota_over25 or not self.quota_under25:
            return None
        raw_o, raw_u = 1 / self.quota_over25, 1 / self.quota_under25
        return raw_o / (raw_o + raw_u)


# ===========================================================================
# MODELLO DI POISSON + CALIBRAZIONE SULLE QUOTE
# ===========================================================================

def lambdas_da_statistiche(casa: StatisticheSquadra, fuori: StatisticheSquadra):
    """λ base dal modello attacco/difesa (Maher, 1982)."""
    att_casa = casa.gol_fatti_media / MEDIA_GOL_CASA_LEGA
    dif_fuori = fuori.gol_subiti_media / MEDIA_GOL_CASA_LEGA
    lam_casa = MEDIA_GOL_CASA_LEGA * att_casa * dif_fuori

    att_fuori = fuori.gol_fatti_media / MEDIA_GOL_FUORI_LEGA
    dif_casa = casa.gol_subiti_media / MEDIA_GOL_FUORI_LEGA
    lam_fuori = MEDIA_GOL_FUORI_LEGA * att_fuori * dif_casa

    return (float(np.clip(lam_casa, 0.2, 4.5)),
            float(np.clip(lam_fuori, 0.2, 4.5)))


def _totale_gol_da_over25(p_over: float) -> float:
    """Inverte la Poisson: λ totale tale che P(gol>=3) = p_over."""
    return brentq(lambda t: (1 - poisson.cdf(2, t)) - p_over, 0.3, 8.0)


def _supremazia_da_1x2(p1: float, p2: float, lam_tot: float) -> float:
    """Differenza λ_casa-λ_fuori coerente con l'asimmetria 1X2 (Skellam)."""
    target = p1 - p2

    def f(d: float) -> float:
        lam_c, lam_f = (lam_tot + d) / 2, (lam_tot - d) / 2
        p_casa = 1 - skellam.cdf(0, lam_c, lam_f)
        p_fuori = skellam.cdf(-1, lam_c, lam_f)
        return (p_casa - p_fuori) - target

    lim = lam_tot - 0.1
    return brentq(f, -lim, lim)


def calibra_con_quote(lam_casa: float, lam_fuori: float, quote: QuoteMatch | None):
    """Fonde i λ statistici con quelli impliciti nelle quote di mercato."""
    if quote is None:
        return lam_casa, lam_fuori

    p1, _px, p2 = quote.probabilita_implicite_1x2()
    p_over = quote.probabilita_implicita_over25()
    tot_mercato = (_totale_gol_da_over25(p_over) if p_over is not None
                   else lam_casa + lam_fuori)
    supr = _supremazia_da_1x2(p1, p2, tot_mercato)
    lam_c_mkt = (tot_mercato + supr) / 2
    lam_f_mkt = (tot_mercato - supr) / 2

    w = PESO_QUOTE
    return (float(max((1 - w) * lam_casa + w * lam_c_mkt, 0.1)),
            float(max((1 - w) * lam_fuori + w * lam_f_mkt, 0.1)))


# ===========================================================================
# MONTE CARLO GAMMA-POISSON
# ===========================================================================

@dataclass
class RisultatiSimulazione:
    """Risultati aggregati delle N simulazioni."""
    gol_casa: np.ndarray
    gol_fuori: np.ndarray
    n_iterazioni: int

    @property
    def p_1(self): return float(np.mean(self.gol_casa > self.gol_fuori))
    @property
    def p_x(self): return float(np.mean(self.gol_casa == self.gol_fuori))
    @property
    def p_2(self): return float(np.mean(self.gol_casa < self.gol_fuori))
    @property
    def p_over25(self): return float(np.mean((self.gol_casa + self.gol_fuori) >= 3))
    @property
    def p_under25(self): return 1.0 - self.p_over25
    @property
    def p_goal(self):
        return float(np.mean((self.gol_casa >= 1) & (self.gol_fuori >= 1)))
    @property
    def p_nogoal(self): return 1.0 - self.p_goal

    def top_risultati_esatti(self, n: int = 3):
        """Top-N risultati esatti più frequenti nelle simulazioni."""
        codici = self.gol_casa * 100 + self.gol_fuori
        valori, conteggi = np.unique(codici, return_counts=True)
        ordine = np.argsort(conteggi)[::-1][:n]
        out = []
        for idx in ordine:
            c, f = divmod(int(valori[idx]), 100)
            out.append((f"{c}-{f}", conteggi[idx] / self.n_iterazioni))
        return out

    def distribuzione_gol_totali(self, max_gol: int = 7):
        """Distribuzione dei gol totali (per il grafico)."""
        totali = self.gol_casa + self.gol_fuori
        return {g: float(np.mean(totali == g) if g < max_gol
                         else np.mean(totali >= max_gol))
                for g in range(max_gol + 1)}


def esegui_monte_carlo(lam_casa: float, lam_fuori: float,
                       n_iter: int = N_SIMULAZIONI,
                       k: float = GAMMA_DISPERSIONE_K,
                       seed: int | None = None) -> RisultatiSimulazione:
    """
    A ogni iterazione i λ vengono campionati da una Gamma (incertezza sui
    parametri): la miscela Gamma-Poisson è una Binomiale Negativa, con
    code più realistiche del Poisson puro.
    """
    rng = np.random.default_rng(seed)
    lam_c_i = rng.gamma(shape=lam_casa * k, scale=1.0 / k, size=n_iter)
    lam_f_i = rng.gamma(shape=lam_fuori * k, scale=1.0 / k, size=n_iter)
    return RisultatiSimulazione(rng.poisson(lam_c_i), rng.poisson(lam_f_i), n_iter)


# ===========================================================================
# INTERFACCIA STREAMLIT
# ===========================================================================

st.set_page_config(page_title="Simulatore Calcio", page_icon="⚽")
st.title("⚽ Simulatore Predittivo di Partite")
st.caption("Poisson calibrato sulle quote + Monte Carlo Gamma-Poisson "
           f"({N_SIMULAZIONI:,} iterazioni)".replace(",", "."))

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

if st.button("🎲 Simula la partita", type="primary"):
    st_casa = StatisticheSquadra(casa, gf_c, gs_c, list(forma_c.upper()))
    st_fuori = StatisticheSquadra(fuori, gf_f, gs_f, list(forma_f.upper()))

    # Pipeline: λ statistici -> calibrazione quote -> correzione forma
    lam_c, lam_f = lambdas_da_statistiche(st_casa, st_fuori)
    lam_c, lam_f = calibra_con_quote(lam_c, lam_f, quote)
    lam_c *= st_casa.punteggio_forma()
    lam_f *= st_fuori.punteggio_forma()

    ris = esegui_monte_carlo(lam_c, lam_f)

    st.metric("Gol attesi", f"{casa} {lam_c:.2f} — {lam_f:.2f} {fuori}")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"1 · {casa}", f"{ris.p_1:.1%}")
    m2.metric("X · Pareggio", f"{ris.p_x:.1%}")
    m3.metric(f"2 · {fuori}", f"{ris.p_2:.1%}")

    st.subheader("Top 3 risultati esatti")
    st.table(pd.DataFrame(
        [(s, f"{p:.1%}") for s, p in ris.top_risultati_esatti(3)],
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
