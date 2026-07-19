"""
main.py — Entry point CLI del Simulatore Predittivo di Partite di Calcio.

Flusso:
  1. Input match (competizione, data, squadre)
  2. Recupero dati: API-Football se disponibile la chiave, altrimenti manuale
  3. Recupero quote: The Odds API se disponibile, altrimenti manuale
  4. Calcolo λ (Poisson) -> calibrazione con quote -> correzione forma
  5. Monte Carlo 10.000 iterazioni (Gamma-Poisson)
  6. Report a schermo

Uso:
    python main.py
"""

from __future__ import annotations

import sys

from src import config
from src.data_fetcher import (
    APIFootballClient,
    OddsAPIClient,
    QuoteMatch,
    StatisticheSquadra,
    input_manuale_quote,
    input_manuale_squadra,
)
from src.models import monte_carlo, poisson_model
from src.render import stampa_report


def raccogli_input_match() -> tuple[str, str, str, str]:
    """Chiede all'utente i dati identificativi del match."""
    print("=" * 62)
    print("  SIMULATORE PREDITTIVO DI PARTITE DI CALCIO")
    print("=" * 62)
    competizione = input("\nCompetizione (es. Serie A): ").strip()
    data_match = input("Data del match (es. 2026-08-23): ").strip()
    casa = input("Squadra di CASA: ").strip()
    fuori = input("Squadra FUORI casa: ").strip()
    return competizione, data_match, casa, fuori


def recupera_statistiche(
    competizione: str, casa: str, fuori: str,
) -> tuple[StatisticheSquadra, StatisticheSquadra]:
    """
    Prova l'API se la chiave è configurata; in caso di qualunque errore
    (limiti free tier, squadra non trovata, rete) ripiega sull'input manuale.
    """
    if config.API_FOOTBALL_KEY:
        try:
            league_id = config.LEAGUE_IDS[competizione.lower()]
            stagione = int(input("Stagione (es. 2025): ").strip())
            client = APIFootballClient()
            id_casa = client.cerca_team_id(casa, league_id, stagione)
            id_fuori = client.cerca_team_id(fuori, league_id, stagione)
            st_casa = client.statistiche_squadra(id_casa, league_id, stagione, in_casa=True)
            st_fuori = client.statistiche_squadra(id_fuori, league_id, stagione, in_casa=False)
            print("[OK] Statistiche recuperate da API-Football.")
            return st_casa, st_fuori
        except Exception as e:  # noqa: BLE001 — fallback voluto su qualunque errore
            print(f"[!] API-Football non disponibile ({e}). Passo all'input manuale.")

    return (
        input_manuale_squadra(casa, "in casa"),
        input_manuale_squadra(fuori, "fuori casa"),
    )


def recupera_quote(competizione: str, casa: str, fuori: str) -> QuoteMatch | None:
    """Come sopra: The Odds API se possibile, altrimenti input manuale."""
    if config.ODDS_API_KEY:
        try:
            quote = OddsAPIClient().quote_match(competizione, casa, fuori)
            if quote:
                print("[OK] Quote recuperate da The Odds API.")
                return quote
            print("[!] Match non trovato su The Odds API.")
        except Exception as e:  # noqa: BLE001
            print(f"[!] The Odds API non disponibile ({e}).")
    return input_manuale_quote()


def main() -> int:
    """Orchestrazione completa della simulazione."""
    competizione, data_match, casa, fuori = raccogli_input_match()

    # --- Dati e quote ---
    st_casa, st_fuori = recupera_statistiche(competizione, casa, fuori)
    quote = recupera_quote(competizione, casa, fuori)

    # --- Pipeline del modello ---
    # 1. λ base dal modello attacco/difesa
    lam_c, lam_f = poisson_model.lambdas_da_statistiche(st_casa, st_fuori)
    # 2. Calibrazione con le quote di mercato (se presenti)
    lam_c, lam_f = poisson_model.calibra_con_quote(lam_c, lam_f, quote)
    # 3. Correzione per la forma recente
    lam_c, lam_f = poisson_model.applica_forma(lam_c, lam_f, st_casa, st_fuori)

    # --- Monte Carlo ---
    risultati = monte_carlo.esegui_monte_carlo(lam_c, lam_f)

    # --- Output ---
    stampa_report(casa, fuori, competizione, data_match, lam_c, lam_f, risultati)
    return 0


if __name__ == "__main__":
    sys.exit(main())
