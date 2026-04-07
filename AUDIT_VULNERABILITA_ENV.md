# Audit tecnico: vulnerabilità, naming e variabili ambiente

Data audit: 2026-04-07

## 1) Vulnerabilità / rischi operativi trovati

### 1.1 Force-push nel workflow manuale
- File: `.github/workflows/manual.yml`
- Evidenza: il job esegue `git push origin "$branch" --force`.
- Rischio: sovrascrittura della storia Git del branch target e potenziale perdita di commit.
- Raccomandazione:
  - usare `--force-with-lease` invece di `--force`;
  - limitare i branch aggiornabili a una allowlist;
  - evitare push automatico su `main` se non strettamente necessario.

### 1.2 Token GitHub con permessi write in workflow schedulato
- File: `.github/workflows/manual.yml`
- Evidenza: checkout usa `token: ${{ secrets.GH_TOKEN }}` con `permissions: contents: write`.
- Rischio: in caso di compromissione del job, il token consente scrittura repository.
- Raccomandazione:
  - usare `GITHUB_TOKEN` con permessi minimali quando possibile;
  - mantenere `GH_TOKEN` solo se necessario a push cross-branch;
  - ruotare periodicamente i secret.

## 2) Inconsistenze naming variabili (codice + workflow)

### 2.1 URI MongoDB: `MONGO_URI` vs `MONGODB_URI`
- In codice core era richiesto solo `MONGO_URI`.
- Nei workflow GitHub viene esportato quasi sempre `MONGODB_URI`.
- Impatto: fallback su cache locale/JSON, job non allineati, errori intermittenti.
- Stato: **risolto in questo intervento** su `core/db.py` con supporto a entrambe le chiavi in:
  - `st.secrets`
  - environment
  - file `.env`

### 2.2 Messaggistica operativa non uniforme
- Alcuni script/log parlano di `MONGODB_URI`, altri di `MONGO_URI`.
- Impatto: troubleshooting più lento.
- Stato: **parzialmente allineato** aggiornando il messaggio in `train_pipeline.py` a `MONGO_URI/MONGODB_URI`.

## 3) Inconsistenze naming variabili applicative

### 3.1 Nomi round eterogenei
- Trovati nomi multipli per stesso concetto tra pipeline/workflow:
  - `round_num` (DB/model)
  - `through_round` (CLI training)
  - `from_round` / `to_round` (workflow fetch)
- Rischio: mapping errato nei passaggi tra job e script.
- Raccomandazione:
  - standardizzare semanticamente:
    - `round_num` solo per record singolo,
    - `through_round` per limite superiore training,
    - `from_round`/`to_round` solo in fetch range;
  - documentare questa convenzione in README/WORKFLOW.md.

## 4) Variabili ambiente censite

### 4.1 Variabili usate dal codice
- Mongo:
  - `MONGO_URI` (storico)
  - `MONGODB_URI` (ora supportata)
  - `MONGO_DB`
- Odds API:
  - `THE_ODDS_API_KEY`

### 4.2 Variabili usate dai workflow
- `.github/workflows/fetch_data.yml`
  - `MONGODB_URI`, `THE_ODDS_API_KEY`
- `.github/workflows/retrain.yml`
  - `MONGODB_URI`, `THE_ODDS_API_KEY`, `MONGO_DB`
- `.github/workflows/manual.yml`
  - `GH_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`

## 5) Piano di remediation consigliato

1. **Standard segreto unico (consigliato):** mantenere `MONGODB_URI` in GitHub Secrets.
2. **Compatibilità runtime:** mantenere supporto duale `MONGO_URI` + `MONGODB_URI` (implementato).
3. **Hardening workflow manuale:** sostituire `--force` con `--force-with-lease` e restringere branch target.
4. **Documentazione unica:** aggiornare README e ISTRUZIONI_ESECUZIONE con una policy chiara sulle env.
5. **Guardrail CI:** aggiungere check statico che fallisce se viene introdotta una nuova alias env non documentata.
