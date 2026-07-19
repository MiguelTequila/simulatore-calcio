# ⚽ Simulatore Predittivo di Partite di Calcio

Motore predittivo in Python che stima probabilità 1X2, risultati esatti, Under/Over 2.5 e Goal/No Goal per una partita di calcio, combinando:

1. **Modello di Poisson attacco/difesa** (Maher, 1982) sui dati storici delle squadre;
2. **Calibrazione sulle quote di mercato** — le probabilità implicite 1X2 e O/U 2.5 (de-marginate) riconciliano totale gol e supremazia via distribuzioni di Poisson e Skellam;
3. **Monte Carlo Gamma-Poisson a 10.000 iterazioni** — i λ vengono campionati da una Gamma a ogni iterazione (miscela = Binomiale Negativa), introducendo sovradispersione realistica invece di replicare inutilmente la matrice analitica.

## Struttura del repository

```
simulatore-calcio/
├── main.py                  # Entry point CLI
├── app_streamlit.py         # Interfaccia web opzionale
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── config.py            # Chiavi API, ID leghe, parametri modello
    ├── data_fetcher.py      # API-Football, The Odds API, input manuale
    ├── render.py            # Report CLI + istogramma testuale
    └── models/
        ├── poisson_model.py # λ, calibrazione quote, matrice risultati
        └── monte_carlo.py   # Simulazione vettorizzata Gamma-Poisson
```

## Installazione

```bash
git clone https://github.com/TUO_USERNAME/simulatore-calcio.git
cd simulatore-calcio
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configurazione API (opzionale)

Il simulatore funziona **senza alcuna chiave** in modalità input manuale. Per l'automazione:

```bash
export API_FOOTBALL_KEY="la_tua_chiave"   # https://www.api-football.com (free: ~100 req/giorno)
export ODDS_API_KEY="la_tua_chiave"       # https://the-odds-api.com  (free: 500 req/mese)
```

Le chiavi restano fuori dal codice e da GitHub (lette da variabili d'ambiente, `.env` in `.gitignore`).

## Uso

**CLI:**
```bash
python main.py
```

**Web:**
```bash
streamlit run app_streamlit.py
```

## Esempio di output CLI

```
  ESITO FINALE (1X2)
    1  (Roma           ) :  44.2%
    X  (Pareggio       ) :  27.5%
    2  (Lazio          ) :  28.3%

  TOP 3 RISULTATI ESATTI
    1. 1-1   :  12.1%
    2. 1-0   :  10.4%
    3. 2-1   :   8.9%
```

## Caricamento su GitHub

```bash
cd simulatore-calcio
git init
git add .
git commit -m "Prima versione: Poisson calibrato + Monte Carlo Gamma-Poisson"
git branch -M main
git remote add origin https://github.com/TUO_USERNAME/simulatore-calcio.git
git push -u origin main
```

## Limiti dichiarati (leggili)

- **Il modello non batte i bookmaker.** Usa le loro quote come input: descrive il mercato, non lo sfrutta. Le probabilità sono descrittive, non un consiglio di scommessa.
- Poisson indipendente sottostima leggermente i pareggi a basso punteggio; il campionamento Gamma mitiga ma non elimina il bias.
- Nessuna gestione di infortuni, squalifiche, motivazioni, meteo.
- Il piano gratuito di API-Football limita richieste e stagioni accessibili.

## Licenza

MIT — uso a scopo didattico e di intrattenimento.
