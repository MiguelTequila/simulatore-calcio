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
# DATABASE SQUADRE 2026/27 (inclusi i preliminari Champions/Europa/Conference)
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
    "Liga Portugal": [
        "Arouca", "AVS", "Benfica", "Boavista", "Braga", "Casa Pia", "Estoril",
        "Estrela Amadora", "Famalicao", "FC Porto", "Farense", "Gil Vicente",
        "Guimarães", "Moreirense", "Nacional", "Rio Ave", "Santa Clara",
        "Sporting CP",
    ],
    "Super Lig": [
        "Besiktas", "Fenerbahce", "Galatasaray", "Trabzonspor", "Basaksehir",
        "Kasimpasa", "Sivasspor", "Adana Demirspor",
    ],
    "Scottish Premiership": [
        "Celtic", "Rangers", "Hearts", "Kilmarnock", "St. Mirren", "Aberdeen",
    ],
    "Qualificazioni / Altri Europei": [
        # Svizzera
        "Young Boys", "Lugano", "Servette", "Basilea", "Zurigo",
        # Belgio
        "Club Brugge", "Union SG", "Anderlecht", "Gent", "Genk", "Cercle Brugge",
        # Austria
        "Salisburgo", "Sturm Graz", "LASK", "Rapid Vienna", "Austria Vienna",
        # Grecia
        "PAOK", "AEK Atene", "Olympiacos", "Panathinaikos", "Aris Salonicco",
        # Repubblica Ceca & Slovacchia
        "Sparta Praga", "Slavia Praga", "Viktoria Plzen", "Slovan Bratislava",
        # Ucraina & Polonia
        "Shakhtar Donetsk", "Dynamo Kiev", "Jagiellonia", "Slask Wroclaw", "Legia Varsavia",
        # Croazia, Serbia, Slovenia, Ungheria
        "Dinamo Zagabria", "Rijeka", "Hajduk Spalato", "Stella Rossa", "Partizan",
        "TSC Backa Topola", "NK Celje", "Maribor", "Ferencvaros", "Paks",
        # Danimarca, Norvegia, Svezia
        "FC Copenaghen", "Midtjylland", "Brøndby", "Nordsjælland", "Bodo/Glimt",
        "Molde", "Brann", "Malmö FF", "Elfsborg", "BK Häcken",
        # Cipro, Israele, Romania, Bulgaria
        "APOEL Nicosia", "Paphos", "Maccabi Tel Aviv", "Maccabi Haifa", "FCSB",
        "CFR Cluj", "Ludogorets", "CSKA Sofia",
        # Altre leghe minori / Preliminari UEFA
        "Slovan Bratislava", "Qarabag", "KÍ Klaksvík", "Flora Tallinn",
        "RFS Riga", "Panevezys", "Zalgiris", "Ordabasy", "Pyunik", "Petrocub",
        "Dinamo Minsk", "Lincoln Red Imps", "The New Saints", "Ballkani",
        "Egnatia", "Hamrun Spartans", "Differdange", "Vikingur Reykjavik",
        "Larne", "Decic", "FC Santa Coloma", "Virtus",
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
_TUTTI_I_CLUB = ["Serie A", "Premier League", "Liga BBVA", "Bundesliga",
                 "Ligue 1", "Eredivisie", "Liga Portugal", "Super Lig",
                 "Scottish Premiership", "Qualificazioni / Altri Europei"]

COMPETIZIONI = {
    "Serie A": ["Serie A"],
    "Premier League": ["Premier League"],
    "Liga BBVA": ["Liga BBVA"],
    "Bundesliga": ["Bundesliga"],
    "Ligue 1": ["Ligue 1"],
    "Eredivisie": ["Eredivisie"],
    "Liga Portugal": ["Liga Portugal"],
    "Champions League": _TUTTI_I_CLUB,
    "Europa League": _TUTTI_I_CLUB,
    "Conference League": _TUTTI_I_CLUB,
    "Coppa Italia": ["Serie A"],
    "Copa del Rey": ["Liga BBVA"],
    "FA Cup": ["Premier League"],
    "DFB Pokal": ["Bundesliga"],
    "Coppa di Francia": ["Ligue 1"],
    "KNVB Beker": ["Eredivisie"],
    "Taça de Portugal": ["Liga Portugal"],
    "UEFA Nations League": ["Nazionali"],
    "Amichevole": _TUTTI_I_CLUB + ["Nazionali"],
}

ALTRA = "✏️ Altra squadra…"
ALTRA_COMP = "✏️ Altra competizione…"


def opzioni_squadre(competizione: str) -> list:
    """
    Elenco squadre per la competizione scelta + voce a inserimento libero.
    Per una competizione non in elenco (inserita a mano) si mostra
    l'unione di tutti i club e le nazionali in archivio.
    """
    liste = COMPETIZIONI.get(competizione)
    if liste is None:
        liste = _TUTTI_I_CLUB + ["Nazionali"]
    squadre = []
    for lista in liste:
        squadre.extend(SQUADRE[lista])
    return sorted(set(squadre)) + [ALTRA]


# ===========================================================================
# REGISTRO PREVISIONI E CALIBRAZIONE
# ===========================================================================

COLONNE_REGISTRO = [
    "data", "competizione", "casa", "fuori",
    "p1", "px", "p2", "p_over25", "p_gg", "top_risultato",
    "p1_mkt", "px_mkt", "p2_mkt", "p_over_mkt", "p_gg_mkt",
    "gol_casa_reale", "gol_fuori_reale",
]


def registro_vuoto() -> pd.DataFrame:
    """DataFrame vuoto con le colonne del registro."""
    return pd.DataFrame(columns=COLONNE_REGISTRO)


def _esiti_reali(gc: float, gf: float) -> dict:
    """Converte un risultato reale negli esiti binari dei vari mercati."""
    return {
        "o1": 1.0 if gc > gf else 0.0,
        "ox": 1.0 if gc == gf else 0.0,
        "o2": 1.0 if gc < gf else 0.0,
        "o_over": 1.0 if (gc + gf) >= 3 else 0.0,
        "o_gg": 1.0 if (gc >= 1 and gf >= 1) else 0.0,
    }


def calcola_metriche(df: pd.DataFrame) -> dict | None:
    """
    Brier score di modello e mercato sulle partite con risultato
    registrato. Restituisce None se non ce ne sono.
    """
    d = df.dropna(subset=["gol_casa_reale", "gol_fuori_reale"])
    if d.empty:
        return None

    b_mod_1x2, b_mkt_1x2 = [], []
    b_mod_over, b_mkt_over = [], []
    b_mod_gg, b_mkt_gg = [], []
    centri_top = 0

    for _, r in d.iterrows():
        gc, gf = float(r["gol_casa_reale"]), float(r["gol_fuori_reale"])
        o = _esiti_reali(gc, gf)

        b_mod_1x2.append((r["p1"] - o["o1"]) ** 2 + (r["px"] - o["ox"]) ** 2
                         + (r["p2"] - o["o2"]) ** 2)
        b_mod_over.append((r["p_over25"] - o["o_over"]) ** 2)
        b_mod_gg.append((r["p_gg"] - o["o_gg"]) ** 2)

        # Mercato: solo dove le quote erano state inserite
        if pd.notna(r.get("p1_mkt")):
            b_mkt_1x2.append((r["p1_mkt"] - o["o1"]) ** 2
                             + (r["px_mkt"] - o["ox"]) ** 2
                             + (r["p2_mkt"] - o["o2"]) ** 2)
        if pd.notna(r.get("p_over_mkt")):
            b_mkt_over.append((r["p_over_mkt"] - o["o_over"]) ** 2)
        if pd.notna(r.get("p_gg_mkt")):
            b_mkt_gg.append((r["p_gg_mkt"] - o["o_gg"]) ** 2)

        if str(r.get("top_risultato", "")) == f"{int(gc)}-{int(gf)}":
            centri_top += 1

    med = lambda x: float(np.mean(x)) if x else None  # noqa: E731
    return {
        "n": len(d),
        "brier_1x2": med(b_mod_1x2), "brier_1x2_mkt": med(b_mkt_1x2),
        "brier_over": med(b_mod_over), "brier_over_mkt": med(b_mkt_over),
        "brier_gg": med(b_mod_gg), "brier_gg_mkt": med(b_mkt_gg),
        "centri_top": centri_top,
        "perc_centri_top": centri_top / len(d),
    }


def tabella_calibrazione(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Il "test del meteorologo": raggruppa TUTTE le probabilità annunciate
    (1, X, 2, Over, GG) in fasce e confronta la media annunciata con la
    frequenza réellement osservata. Se il modello è calibrato, le due
    colonne si somigliano.
    """
    d = df.dropna(subset=["gol_casa_reale", "gol_fuori_reale"])
    if d.empty:
        return None

    coppie = []  # (probabilità annunciata, esito 0/1)
    for _, r in d.iterrows():
        gc, gf = float(r["gol_casa_reale"]), float(r["gol_fuori_reale"])
        o = _esiti_reali(gc, gf)
        coppie += [
            (r["p1"], o["o1"]), (r["px"], o["ox"]), (r["p2"], o["o2"]),
            (r["p_over25"], o["o_over"]), (r["p_gg"], o["o_gg"]),
        ]

    cp = pd.DataFrame(coppie, columns=["p", "esito"])
    fasce = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    etichette = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    cp["fascia"] = pd.cut(cp["p"], bins=fasce, labels=etichette,
                          include_lowest=True)
    g = cp.groupby("fascia", observed=False).agg(
        casi=("p", "size"), annunciata=("p", "mean"),
        osservata=("esito", "mean"))
    g = g[g["casi"] > 0].reset_index()
    g["annunciata"] = (g["annunciata"] * 100).round(1)
    g["osservata"] = (g["osservata"] * 100).round(1)
    g.columns = ["Fascia", "Casi", "Annunciata %", "Osservata %"]
    return g


def media_inizio_stagione(scorsa: float, corrente: float, giornate: int,
                          neopromossa: bool, gol_subiti: bool,
                          media_lega: float = 1.35) -> float:
    """
    Media gol da usare a inizio stagione, quando le partite giocate
    sono poche e la media corrente è ancora rumore.
    """
    if neopromossa:
        base = 1.6 if gol_subiti else 1.0
    else:
        base = 0.6 * scorsa + 0.4 * media_lega
    peso_corrente = min(max(giornate, 0) / 10.0, 1.0)
    return round(peso_corrente * corrente + (1 - peso_corrente) * base, 2)


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
Competizione non in elenco (campionati esteri, Serie B, Mondiali…)? Usa
*✏️ Altra competizione* e scrivi il nome: le squadre mostrate diventano
tutte quelle in archivio. Squadra comunque assente (coppe nazionali,
club minori)? Usa *✏️ Altra squadra* e digitala a mano.

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

**7. Registro previsioni (in fondo alla pagina)** — è il quaderno che
trasforma le sensazioni in misure:
- dopo ogni simulazione premi *💾 Salva questa previsione*;
- a partita giocata torna qui e registra il risultato reale;
- **scarica sempre il CSV prima di chiudere la scheda** (l'app gira nel
  browser: senza file scaricato le previsioni si perdono), e ricaricalo
  la volta dopo per continuare ad accumulare;
- da ~30 partite il *Brier score* dice quanto è affidabile il modello
  (più basso = meglio) e — dato più importante — se batte le quote nude.
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

_scelta_comp = st.selectbox("Competizione",
                            list(COMPETIZIONI.keys()) + [ALTRA_COMP])
if _scelta_comp == ALTRA_COMP:
    # Competizione libera: le squadre mostrate sono l'unione di tutto
    competizione = st.text_input(
        "Nome della competizione",
        placeholder="es. Serie B, MLS, Mondiali…").strip()
    competizione = competizione or "Altra competizione"
    st.caption("Elenco squadre esteso a tutti i club e le nazionali in "
               "archivio. Se la squadra non c'è, usa *✏️ Altra squadra*.")
else:
    competizione = _scelta_comp

if competizione == "Amichevole":
    st.caption("⚠️ Amichevole: turnover e motivazioni ballerine rendono "
               "statistiche e forma meno affidabili. Valuta il campo neutro.")

col_a, col_b = st.columns(2)
with col_a:
    casa = me_squadra = scegli_squadra("Squadra di casa", competizione, "casa", default=0)
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
    st_casa = StatisticheSquadra(casa, gf_c, gs_c, list(forma_c.upper())[-5:])
    st_fuori = StatisticheSquadra(fuori, gf_f, gs_f, list(forma_f.upper())[-5:])

    lam_c, lam_f = lambdas_da_statistiche(st_casa, st_fuori, campo_neutro)
    lam_c, lam_f = calibra_con_quote(lam_c, lam_f, quote)
    lam_c *= st_casa.punteggio_forma()
    lam_f *= st_fuori.punteggio_forma()

    st.session_state.ultima = {
        "ris": esegui_monte_carlo(lam_c, lam_f),
        "lam_c": lam_c, "lam_f": lam_f,
        "casa": casa, "fuori": fuori, "competizione": competizione,
        "data": str(data_match), "quote": quote,
        "eliminazione": eliminazione, "ritorno": ritorno,
        "andata": (gol_andata_casa, gol_andata_fuori),
    }
    st.session_state.salvata = False

u = st.session_state.get("ultima")
if u:
    ris, lam_c, lam_f = u["ris"], u["lam_c"], u["lam_f"]
    nome_c, nome_f, quote_u = u["casa"], u["fuori"], u["quote"]

    st.metric("Gol attesi", f"{nome_c} {lam_c:.2f} — {lam_f:.2f} {nome_f}")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"1 · {nome_c}", f"{ris.p_1:.1%}")
    m2.metric("X · Pareggio", f"{ris.p_x:.1%}")
    m3.metric(f"2 · {nome_f}", f"{ris.p_2:.1%}")

    st.subheader("Top 3 risultati esatti")
    top3 = ris.top_risultati_esatti(3)
    st.table(pd.DataFrame([(s, f"{p:.1%}") for s, p in top3],
                          columns=["Risultato", "Probabilità"]))

    m4, m5, m6, m7 = st.columns(4)
    m4.metric("Over 2.5", f"{ris.p_over25:.1%}")
    m5.metric("Under 2.5", f"{ris.p_under25:.1%}")
    m6.metric("Goal (GG)", f"{ris.p_goal:.1%}")
    m7.metric("NoGoal (NG)", f"{ris.p_nogoal:.1%}")

    gg_mercato = quote_u.probabilita_implicita_gg() if quote_u else None
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

    if u["eliminazione"]:
        st.subheader("⚔️ Passaggio del turno"
                     + (" (aggregato)" if u["ritorno"] else ""))
        elim = risolvi_eliminazione(ris, lam_c, lam_f, andata=u["andata"])
        if u["ritorno"]:
            st.caption(f"Andata: {nome_c} {u['andata'][0]} — "
                       f"{u['andata'][1]} {nome_f}")
        e1, e2 = st.columns(2)
        e1.metric(f"Passa {nome_c}", f"{elim['passa_casa']:.1%}")
        e2.metric(f"Passa {nome_f}", f"{elim['passa_fuori']:.1%}")
        st.caption(
            f"Pareggio ai 90': {elim['p_supplementari']:.1%} dei casi. "
            f"Ripartizione {nome_c}: 90' {elim['casa_90']:.1%} · "
            f"suppl. {elim['casa_sup']:.1%} · rigori {elim['casa_rig']:.1%}. "
            f"{nome_f}: 90' {elim['fuori_90']:.1%} · "
            f"suppl. {elim['fuori_sup']:.1%} · rigori {elim['fuori_rig']:.1%}."
        )

    if st.session_state.get("salvata"):
        st.success("✔️ Previsione salvata nel registro (in fondo alla pagina).")
    elif st.button("💾 Salva questa previsione nel registro"):
        p1m = pxm = p2m = pom = pgm = np.nan
        if quote_u:
            p1m, pxm, p2m = quote_u.probabilita_implicite_1x2()
            pom = quote_u.probabilita_implicita_over25() or np.nan
            pgm = quote_u.probabilita_implicita_gg() or np.nan
        riga = {
            "data": u["data"], "competizione": u["competizione"],
            "casa": nome_c, "fuori": nome_f,
            "p1": ris.p_1, "px": ris.p_x, "p2": ris.p_2,
            "p_over25": ris.p_over25, "p_gg": ris.p_goal,
            "top_risultato": top3[0][0],
            "p1_mkt": p1m, "px_mkt": pxm, "p2_mkt": p2m,
            "p_over_mkt": pom, "p_gg_mkt": pgm,
            "gol_casa_reale": np.nan, "gol_fuori_reale": np.nan,
        }
        st.session_state.registro = pd.concat(
            [st.session_state.get("registro", registro_vuoto()),
             pd.DataFrame([riga])], ignore_index=True)
        st.session_state.salvata = True
        st.rerun()

    st.info("Probabilità descrittive del mercato, non consigli di scommessa. "
            "Un modello calibrato sulle quote non può batterle.")


# ===========================================================================
# SEZIONE REGISTRO PREVISIONI
# ===========================================================================
st.divider()
st.subheader("📒 Registro previsioni")

if "registro" not in st.session_state:
    st.session_state.registro = registro_vuoto()

st.warning(
    "⚠️ L'app gira nel tuo browser: il registro vive solo in questa scheda. "
    "**Scarica il CSV prima di chiudere**, e ricaricalo la volta dopo per "
    "continuare ad accumulare partite.")

caricato = st.file_uploader("Ricarica un registro salvato (.csv)", type="csv")
if caricato is not None and not st.session_state.get("caricato_fatto"):
    try:
        df_in = pd.read_csv(caricato)
        mancanti = [c for c in COLONNE_REGISTRO if c not in df_in.columns]
        if mancanti:
            st.error(f"CSV non valido, colonne mancanti: {', '.join(mancanti)}")
        else:
            st.session_state.registro = pd.concat(
                [st.session_state.registro, df_in[COLONNE_REGISTRO]],
                ignore_index=True).drop_duplicates(
                    subset=["data", "casa", "fuori"], keep="last")
            st.session_state.caricato_fatto = True
            st.success(f"Caricate {len(df_in)} previsioni.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Impossibile leggere il file: {e}")

reg = st.session_state.registro

if reg.empty:
    st.caption("Nessuna previsione salvata. Simula una partita e premi "
               "«💾 Salva questa previsione nel registro».")
else:
    da_completare = reg[reg["gol_casa_reale"].isna()]
    if not da_completare.empty:
        st.markdown("**Registra il risultato di una partita giocata**")
        etichette = {
            f"{r['data']} · {r['casa']} vs {r['fuori']}": i
            for i, r in da_completare.iterrows()
        }
        scelta = st.selectbox("Partita in attesa di risultato",
                              list(etichette.keys()))
        g1, g2, g3 = st.columns([2, 2, 3])
        gc_reale = g1.number_input("Gol casa", 0, 20, 0, key="res_c")
        gf_reale = g2.number_input("Gol fuori", 0, 20, 0, key="res_f")
        if g3.button("✅ Registra risultato"):
            idx = etichette[scelta]
            st.session_state.registro.loc[idx, "gol_casa_reale"] = gc_reale
            st.session_state.registro.loc[idx, "gol_fuori_reale"] = gf_reale
            st.rerun()

    vista = reg.copy()
    for col in ["p1", "px", "p2", "p_over25", "p_gg"]:
        vista[col] = (vista[col].astype(float) * 100).round(1)
    st.dataframe(
        vista[["data", "casa", "fuori", "p1", "px", "p2", "p_over25",
               "p_gg", "top_risultato", "gol_casa_reale", "gol_fuori_reale"]],
        use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Scarica il registro (.csv)",
        data=reg.to_csv(index=False).encode("utf-8"),
        file_name="registro_previsioni.csv", mime="text/csv")

    met = calcola_metriche(reg)
    if met is None:
        st.caption("Nessun risultato registrato ancora: le metriche "
                   "compaiono appena inserisci il primo.")
    else:
        st.markdown("---")
        st.markdown(f"**Punteggio su {met['n']} partite concluse**")
        if met["n"] < 30:
            st.caption(f"⚠️ Solo {met['n']} partite: i numeri qui sotto sono "
                       "ancora rumore. Servono almeno 30 partite per una "
                       "lettura indicativa, 50+ per fidarsi.")

        st.caption("Brier score: più BASSO è, meglio è. Riferimenti: "
                   "sparare a caso vale 0.667 sull'1X2 e 0.250 sui mercati "
                   "binari (Over, GG).")
        b1, b2, b3 = st.columns(3)
        b1.metric("Brier 1X2", f"{met['brier_1x2']:.3f}",
                  delta=(None if met["brier_1x2_mkt"] is None else
                         f"{met['brier_1x2'] - met['brier_1x2_mkt']:+.3f} vs "
                         "mercato"), delta_color="inverse")
        b2.metric("Brier Over 2.5", f"{met['brier_over']:.3f}",
                  delta=(None if met["brier_over_mkt"] is None else
                         f"{met['brier_over'] - met['brier_over_mkt']:+.3f} vs "
                         "mercato"), delta_color="inverse")
        b3.metric("Brier GG", f"{met['brier_gg']:.3f}",
                  delta=(None if met["brier_gg_mkt"] is None else
                         f"{met['brier_gg'] - met['brier_gg_mkt']:+.3f} vs "
                         "mercato"), delta_color="inverse")
        st.caption(
            "Il confronto «vs mercato» è il verdetto vero: valore NEGATIVO "
            "(verde) = il modello batte le quote nude su quelle partite; "
            "POSITIVO (rosso) = le quote da sole facevano meglio, e la "
            "componente statistica sta togliendo invece di aggiungere.")

        st.caption(f"Risultato esatto più probabile azzeccato "
                   f"{met['centri_top']} volte su {met['n']} "
                   f"({met['perc_centri_top']:.0%}). Riferimento: un modello "
                   "sano ci prende attorno al 10-13%.")

        tab = tabella_calibrazione(reg)
        if tab is not None:
            st.markdown("**Test del meteorologo (calibrazione)**")
            st.caption("Quando il modello annuncia una certa probabilità, "
                       "l'evento accade davvero con quella frequenza? Le due "
                       "colonne devono somigliarsi.")
            st.dataframe(tab, use_container_width=True, hide_index=True)
