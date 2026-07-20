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
    quota_gg: float | None = None    # Goal: entrambe segnano
    quota_ng: float | None = None    # NoGoal

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

    def probabilita_implicita_gg(self):
        """
        Probabilità de-marginata del Goal (GG). NON entra nella
        calibrazione — i due λ sono già fissati da 1X2 e O/U 2.5 — ma
        serve come CONFRONTO diagnostico col GG calcolato dal modello.
        """
        if not self.quota_gg or not self.quota_ng:
            return None
        raw_g, raw_n = 1 / self.quota_gg, 1 / self.quota_ng
        return raw_g / (raw_g + raw_n)


# ===========================================================================
# MODELLO DI POISSON + CALIBRAZIONE SULLE QUOTE
# ===========================================================================

def lambdas_da_statistiche(casa: StatisticheSquadra, fuori: StatisticheSquadra,
                           campo_neutro: bool = False):
    """
    λ base dal modello attacco/difesa (Maher, 1982).

    Campo neutro: si usa la STESSA media di riferimento per entrambe
    (media tra casa e trasferta di lega) e nessuna squadra riceve il
    vantaggio del fattore campo. Corretto per finali, tornei, gare
    in sede unica.
    """
    if campo_neutro:
        m = (MEDIA_GOL_CASA_LEGA + MEDIA_GOL_FUORI_LEGA) / 2  # 1.35
        lam_casa = m * (casa.gol_fatti_media / m) * (fuori.gol_subiti_media / m)
        lam_fuori = m * (fuori.gol_fatti_media / m) * (casa.gol_subiti_media / m)
    else:
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


def risolvi_eliminazione(ris: RisultatiSimulazione, lam_casa: float,
                         lam_fuori: float, andata: tuple = (0, 0),
                         seed: int | None = None):
    """
    Per gare a eliminazione diretta: chi passa il turno?

    `andata` = (gol squadra ora in casa segnati all'andata, gol squadra
    ora fuori). Con (0, 0) equivale a una gara secca. Il confronto è sul
    punteggio AGGREGATO (regola dei gol in trasferta abolita, UEFA 2021):
    se l'aggregato è pari dopo i 90' del ritorno -> supplementari
    (λ ridotti a 1/3, cioè 30' su 90') e, se ancora pari, rigori (50/50).
    """
    rng = np.random.default_rng(seed)
    n = ris.n_iterazioni
    and_c, and_f = andata

    # Punteggi aggregati (andata fissa + ritorno simulato)
    agg_c = ris.gol_casa + and_c
    agg_f = ris.gol_fuori + and_f

    vince_c_90 = agg_c > agg_f
    vince_f_90 = agg_c < agg_f
    pari_90 = agg_c == agg_f
    n_pari = int(pari_90.sum())

    # Supplementari: 30 minuti -> λ/3
    sup_c = rng.poisson(lam_casa / 3.0, n_pari)
    sup_f = rng.poisson(lam_fuori / 3.0, n_pari)
    vince_c_sup = sup_c > sup_f
    vince_f_sup = sup_c < sup_f
    pari_sup = sup_c == sup_f

    # Rigori: moneta equa
    rigori_casa = rng.random(int(pari_sup.sum())) < 0.5

    p_c_90 = vince_c_90.sum() / n
    p_f_90 = vince_f_90.sum() / n
    p_c_sup = vince_c_sup.sum() / n
    p_f_sup = vince_f_sup.sum() / n
    p_c_rig = rigori_casa.sum() / n
    p_f_rig = (len(rigori_casa) - rigori_casa.sum()) / n

    return {
        "passa_casa": float(p_c_90 + p_c_sup + p_c_rig),
        "passa_fuori": float(p_f_90 + p_f_sup + p_f_rig),
        "casa_90": float(p_c_90), "fuori_90": float(p_f_90),
        "casa_sup": float(p_c_sup), "fuori_sup": float(p_f_sup),
        "casa_rig": float(p_c_rig), "fuori_rig": float(p_f_rig),
        "p_supplementari": float(pari_90.sum() / n),
    }


# ===========================================================================
# DATABASE SQUADRE 2026/27 (verificato luglio 2026 — aggiornare ogni estate)
# Per aggiungere/correggere una squadra basta modificare queste liste.
# ===========================================================================

SQUADRE = {
    "Serie A": [
        "Atalanta", "Bologna", "Cagliari", "Como", "Fiorentina", "Frosinone",
        "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza",
        "Napoli", "Parma", "Roma", "Sassuolo", "Torino", "Udinese", "Venezia",
    ],
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Coventry City", "Crystal Palace", "Everton", "Fulham",
        "Hull City", "Ipswich Town", "Leeds United", "Liverpool",
        "Manchester City", "Manchester United", "Newcastle",
        "Nottingham Forest", "Sunderland", "Tottenham",
    ],
    "Liga BBVA": [
        "Alavés", "Athletic Club", "Atlético Madrid", "Barcellona", "Betis",
        "Celta Vigo", "Deportivo La Coruña", "Elche", "Espanyol", "Getafe",
        "Levante", "Málaga", "Osasuna", "Racing Santander", "Rayo Vallecano",
        "Real Madrid", "Real Sociedad", "Siviglia", "Valencia", "Villarreal",
    ],
    "Bundesliga": [
        "Augsburg", "Bayer Leverkusen", "Bayern Monaco", "Borussia Dortmund",
        "Borussia M'gladbach", "Colonia", "Eintracht Francoforte",
        "Elversberg", "Friburgo", "Amburgo", "Hoffenheim", "Lipsia",
        "Mainz", "Paderborn", "Schalke 04", "Stoccarda", "Union Berlino",
        "Werder Brema",
    ],
    "Ligue 1": [
        "Angers", "Auxerre", "Brest", "Le Havre", "Le Mans", "Lens", "Lille",
        "Lorient", "Lione", "Marsiglia", "Monaco", "Nizza", "Paris FC",
        "PSG", "Rennes", "Strasburgo", "Tolosa", "Troyes",
    ],
    "Eredivisie": [
        "ADO Den Haag", "Ajax", "AZ Alkmaar", "Cambuur", "Excelsior",
        "Feyenoord", "Fortuna Sittard", "Go Ahead Eagles", "Groningen",
        "Heerenveen", "NEC Nijmegen", "PEC Zwolle", "PSV", "Sparta Rotterdam",
        "Telstar", "Twente", "Utrecht", "Willem II",
    ],
    "Nazionali": [
        "Italia", "Spagna", "Francia", "Germania", "Inghilterra", "Olanda",
        "Portogallo", "Belgio", "Croazia", "Danimarca", "Svizzera", "Austria",
        "Polonia", "Serbia", "Turchia", "Ucraina", "Scozia", "Galles",
        "Norvegia", "Svezia", "Grecia", "Repubblica Ceca", "Ungheria",
        "Romania", "Slovenia", "Slovacchia", "Albania", "Georgia", "Irlanda",
        "Argentina", "Brasile", "Uruguay", "Colombia", "Messico", "USA",
        "Giappone", "Marocco",
    ],
}

# Ogni competizione -> da quali liste pescare le squadre.
# Le coppe nazionali includono anche club di serie minori: per quelli
# c'è sempre la voce "Altra squadra" col campo libero.
_TUTTI_I_CLUB = ["Serie A", "Premier League", "Liga BBVA", "Bundesliga",
                 "Ligue 1", "Eredivisie"]
COMPETIZIONI = {
    "Serie A": ["Serie A"],
    "Premier League": ["Premier League"],
    "Liga BBVA": ["Liga BBVA"],
    "Bundesliga": ["Bundesliga"],
    "Ligue 1": ["Ligue 1"],
    "Eredivisie": ["Eredivisie"],
    "Champions League": _TUTTI_I_CLUB,
    "Europa League": _TUTTI_I_CLUB,
    "Conference League": _TUTTI_I_CLUB,
    "Coppa Italia": ["Serie A"],
    "Copa del Rey": ["Liga BBVA"],
    "FA Cup": ["Premier League"],
    "DFB Pokal": ["Bundesliga"],
    "Coppa di Francia": ["Ligue 1"],
    "KNVB Beker": ["Eredivisie"],
    "UEFA Nations League": ["Nazionali"],
    "Amichevole": _TUTTI_I_CLUB + ["Nazionali"],
}

ALTRA = "✏️ Altra squadra…"


def media_inizio_stagione(scorsa: float, corrente: float, giornate: int,
                          neopromossa: bool, gol_subiti: bool,
                          media_lega: float = 1.35) -> float:
    """
    Media gol da usare a inizio stagione, quando le partite giocate
    sono poche e la media corrente è ancora rumore.

    Logica:
      - Base = stagione scorsa "tirata" verso la media di lega
        (0.6 x scorsa + 0.4 x 1.35), perché mercato e cambi allenatore
        erodono le prestazioni passate.
      - Per le NEOPROMOSSE la stagione scorsa (categoria inferiore) non
        vale nulla: base = profilo storico della neopromossa tipo
        (~1.0 gol fatti, ~1.6 subiti).
      - Il peso della stagione corrente cresce linearmente con le
        giornate giocate: 0 giornate = solo base, 10+ giornate = solo
        stagione corrente.
    """
    if neopromossa:
        base = 1.6 if gol_subiti else 1.0
    else:
        base = 0.6 * scorsa + 0.4 * media_lega
    peso_corrente = min(max(giornate, 0) / 10.0, 1.0)
    return round(peso_corrente * corrente + (1 - peso_corrente) * base, 2)


def opzioni_squadre(competizione: str) -> list:
    """Elenco squadre per la competizione scelta + voce a inserimento libero."""
    squadre = []
    for lista in COMPETIZIONI.get(competizione, []):
        squadre.extend(SQUADRE[lista])
    return sorted(set(squadre)) + [ALTRA]


def scegli_squadra(etichetta: str, competizione: str, chiave: str, default: int = 0):
    """
    Selectbox con ricerca integrata (digita per filtrare) + fallback a
    campo libero per squadre non in elenco.
    """
    opzioni = opzioni_squadre(competizione)
    scelta = st.selectbox(etichetta, opzioni, index=min(default, len(opzioni) - 1),
                          key=f"sel_{chiave}")
    if scelta == ALTRA:
        return st.text_input(f"Nome squadra ({etichetta.lower()})",
                             key=f"txt_{chiave}").strip() or "Squadra"
    return scelta


# ===========================================================================
# INTERFACCIA STREAMLIT
# ===========================================================================

st.set_page_config(page_title="Simulatore Calcio", page_icon="⚽")
st.title("⚽ Simulatore Predittivo di Partite")
st.caption("Poisson calibrato sulle quote + Monte Carlo Gamma-Poisson "
           f"({N_SIMULAZIONI:,} iterazioni)".replace(",", "."))

# ---------------------------------------------------------------------------
# ISTRUZIONI D'USO
# ---------------------------------------------------------------------------
with st.expander("📖 Istruzioni: come si usa"):
    st.markdown("""
**1. Scegli la competizione** — la tendina delle squadre si adatta da sola.
Squadra non in elenco (coppe nazionali, club minori)? Usa *✏️ Altra squadra*.

**2. Inserisci le medie gol** (campi "Gol fatti/subiti a partita"):
- **Dove trovarle:** *soccerstats.com* → campionato → tab *Home & Away
  tables* (medie casa e trasferta per squadra); in alternativa *FBref*
  o chiedendo a un'AI le medie delle ultime 10 partite ufficiali.
- **Partita normale:** medie IN CASA per la squadra di casa, IN
  TRASFERTA per l'ospite.
- **Campo neutro attivo:** medie COMPLESSIVE (tutte le partite) per
  entrambe.
- **Prime ~10 giornate di campionato:** non usare le medie grezze
  (troppo rumore) — apri il *🧮 Calcolatore inizio stagione* qui sotto
  e copia i valori che ti restituisce.

**3. Forma (ultime 5, es. WWDLW)** — solo partite ufficiali, MAI
amichevoli di club. Prima della 5ª giornata lasciala **vuota**: campo
vuoto = effetto neutro, ed è corretto così. Per le nazionali le
amichevoli contro pari livello si possono contare.

**4. Interruttori:**
- *🏟️ Campo neutro* — finali e sedi uniche: toglie il vantaggio casa.
- *⚔️ Eliminazione diretta* — aggiunge supplementari, rigori e la
  probabilità di passare il turno.
- *🔁 Gara di ritorno* — inserisci il risultato dell'andata: il
  passaggio turno è calcolato sull'aggregato (gol in trasferta aboliti).

**5. Quote bookmaker** — inseriscile sempre se le hai: pesano il 70%
del modello e correggono i tuoi input imprecisi. Copiale da qualunque
bookmaker in formato decimale.

**6. Leggi l'output per quello che è** — probabilità descrittive del
mercato. Il risultato esatto più probabile esce comunque ~1 volta su 8:
un centro non prova che il modello è buono, un buco non prova che è
rotto. Contano 30+ partite, non una.
""")

# ---------------------------------------------------------------------------
# CALCOLATORE INIZIO STAGIONE
# ---------------------------------------------------------------------------
with st.expander("🧮 Calcolatore inizio stagione (prime ~10 giornate)"):
    st.caption("Miscela stagione scorsa e corrente in base alle giornate "
               "giocate. Compilalo una squadra alla volta e copia i due "
               "valori nei campi delle statistiche.")
    giornate = st.number_input("Giornate di campionato già giocate",
                               0, 15, 0, key="calc_g")
    neopromossa = st.checkbox(
        "Squadra neopromossa",
        key="calc_np",
        help="I numeri della categoria inferiore non valgono: si parte "
             "dal profilo storico della neopromossa tipo (1.0 fatti, "
             "1.6 subiti).")
    c_sx, c_dx = st.columns(2)
    with c_sx:
        st.markdown("**Gol fatti / partita**")
        gf_scorsa = st.number_input("Media stagione scorsa", 0.0, 5.0, 1.5,
                                    0.05, key="calc_gfs",
                                    disabled=neopromossa)
        gf_corr = st.number_input("Media stagione corrente", 0.0, 5.0, 1.5,
                                  0.05, key="calc_gfc",
                                  disabled=(giornate == 0))
    with c_dx:
        st.markdown("**Gol subiti / partita**")
        gs_scorsa = st.number_input("Media stagione scorsa ", 0.0, 5.0, 1.2,
                                    0.05, key="calc_gss",
                                    disabled=neopromossa)
        gs_corr = st.number_input("Media stagione corrente ", 0.0, 5.0, 1.2,
                                  0.05, key="calc_gsc",
                                  disabled=(giornate == 0))
    v_gf = media_inizio_stagione(gf_scorsa, gf_corr, giornate,
                                 neopromossa, gol_subiti=False)
    v_gs = media_inizio_stagione(gs_scorsa, gs_corr, giornate,
                                 neopromossa, gol_subiti=True)
    r1, r2 = st.columns(2)
    r1.metric("→ Gol fatti da inserire", f"{v_gf:.2f}")
    r2.metric("→ Gol subiti da inserire", f"{v_gs:.2f}")
    if giornate >= 10:
        st.caption("Con 10+ giornate il calcolatore restituisce la "
                   "stagione corrente pura: puoi smettere di usarlo.")

competizione = st.selectbox("Competizione", list(COMPETIZIONI.keys()))
if competizione == "Amichevole":
    st.caption("⚠️ Amichevole: turnover e motivazioni ballerine rendono "
               "statistiche e forma meno affidabili. Valuta il campo neutro.")

col_a, col_b = st.columns(2)
with col_a:
    casa = scegli_squadra("Squadra di casa", competizione, "casa", default=0)
with col_b:
    data_match = st.date_input("Data del match")
    fuori = scegli_squadra("Squadra fuori casa", competizione, "fuori", default=1)

t1, t2 = st.columns(2)
campo_neutro = t1.checkbox(
    "🏟️ Campo neutro",
    help="Finali e tornei in sede unica: elimina il vantaggio del fattore "
         "campo. Inserisci le medie gol COMPLESSIVE (non solo casa/fuori).")
eliminazione = t2.checkbox(
    "⚔️ Eliminazione diretta",
    help="Se pareggio ai 90': simula supplementari (λ/3) e rigori (50/50) "
         "e calcola chi passa il turno.")

# --- Doppio confronto: questa è la gara di RITORNO ---
gol_andata_casa = gol_andata_fuori = 0
ritorno = False
if eliminazione:
    ritorno = st.checkbox(
        "🔁 Gara di ritorno di un doppio confronto",
        help="Inserisci il risultato dell'andata: il passaggio del turno "
             "verrà calcolato sul punteggio AGGREGATO (regola dei gol in "
             "trasferta abolita dalla UEFA nel 2021).")
    if ritorno:
        r1, r2 = st.columns(2)
        gol_andata_casa = r1.number_input(
            f"Gol di {casa} all'andata", 0, 15, 0, key="and_c")
        gol_andata_fuori = r2.number_input(
            f"Gol di {fuori} all'andata", 0, 15, 0, key="and_f")

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
    q6, q7 = st.columns(2)
    quota_g = q6.number_input(
        "Quota Goal (GG)", 1.01, 10.0, 1.85, 0.05,
        help="Entrambe le squadre segnano. Non calibra il modello (i due "
             "λ sono già fissati da 1X2 e O/U): serve come confronto "
             "diagnostico mercato vs modello.")
    quota_n = q7.number_input("Quota NoGoal (NG)", 1.01, 10.0, 1.85, 0.05)
    quote = QuoteMatch(quota_1, quota_x, quota_2, quota_o, quota_u,
                       quota_g, quota_n)

if st.button("🎲 Simula la partita", type="primary"):
    # [-5:] = solo le ultime 5 lettere della forma, anche se ne scrivi di più
    st_casa = StatisticheSquadra(casa, gf_c, gs_c, list(forma_c.upper())[-5:])
    st_fuori = StatisticheSquadra(fuori, gf_f, gs_f, list(forma_f.upper())[-5:])

    # Pipeline: λ statistici -> calibrazione quote -> correzione forma
    lam_c, lam_f = lambdas_da_statistiche(st_casa, st_fuori, campo_neutro)
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

    # --- Confronto diagnostico GG: modello vs mercato ---
    gg_mercato = quote.probabilita_implicita_gg() if quote else None
    if gg_mercato is not None:
        delta = ris.p_goal - gg_mercato
        st.caption(f"🔍 GG — modello: {ris.p_goal:.1%} · mercato "
                   f"(de-marginato): {gg_mercato:.1%} · scarto: "
                   f"{delta:+.1%}")
        if abs(delta) > 0.05:
            st.warning(
                f"Scarto GG di {abs(delta):.1%} tra modello e mercato: "
                "controlla i dati inseriti (medie gol, quote) oppure il "
                "mercato sta prezzando qualcosa che le tue statistiche "
                "non vedono (infortuni, assenze, turnover).")

    st.subheader("Distribuzione gol totali")
    dist = ris.distribuzione_gol_totali()
    st.bar_chart(pd.DataFrame(
        {"Probabilità": list(dist.values())},
        index=[f"{g}+" if g == max(dist) else str(g) for g in dist],
    ))

    if eliminazione:
        st.subheader("⚔️ Passaggio del turno"
                     + (" (aggregato)" if ritorno else ""))
        elim = risolvi_eliminazione(
            ris, lam_c, lam_f, andata=(gol_andata_casa, gol_andata_fuori))
        if ritorno:
            st.caption(f"Andata: {casa} {gol_andata_casa} — "
                       f"{gol_andata_fuori} {fuori}")
        e1, e2 = st.columns(2)
        e1.metric(f"Passa {casa}", f"{elim['passa_casa']:.1%}")
        e2.metric(f"Passa {fuori}", f"{elim['passa_fuori']:.1%}")
        st.caption(
            f"Pareggio ai 90': {elim['p_supplementari']:.1%} dei casi. "
            f"Ripartizione {casa}: 90' {elim['casa_90']:.1%} · "
            f"suppl. {elim['casa_sup']:.1%} · rigori {elim['casa_rig']:.1%}. "
            f"{fuori}: 90' {elim['fuori_90']:.1%} · "
            f"suppl. {elim['fuori_sup']:.1%} · rigori {elim['fuori_rig']:.1%}."
        )

    st.info("Probabilità descrittive del mercato, non consigli di scommessa. "
            "Un modello calibrato sulle quote non può batterle.")
