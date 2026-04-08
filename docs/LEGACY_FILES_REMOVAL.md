# File da Potenzialmente Rimuovere

**Generato:** 2026-04-08

## Cartelle Vuote o Ridondanti

### `data/racedata/` - VUOTA ✅ RIMUOVERE
- Clone locale di TracingInsights RaceData
- **Stato:** Vuota (i workflow clonano fresh da GitHub)
- **Azione:** Rimuovere cartella

### `data/cache/jolpica/` - CACHE LOCALE
- Cache locale Jolpica API (~100 file JSON)
- **Stato:** Dati già in MongoDB `f1_races`
- **Azione:** Rimuovere dopo verifica che MongoDB è la fonte primaria

## File Loader Legacy

I seguenti loader sono stati sostituiti dai MongoDB loader:

| File | Uso Attuale | Note |
|------|-------------|------|
| `f1_predictor/data/loader_jolpica.py` | Test only | Da rimuovere dopo migrazione completa |
| `f1_predictor/data/loader_tracinginsights.py` | Non usato | Da rimuovere |
| `f1_predictor/data/loader_kaggle.py` | Non usato | Da rimuovere |

### File da Mantenere

| File | Uso | Motivo |
|------|-----|--------|
| `f1_predictor/data/loader_odds.py` | OddsLoader | Per Layer 4 (ancora necessario) |

## Script Import (ora in GitHub Actions)

| Script | Workflow Equivalente | Status |
|--------|---------------------|--------|
| `scripts/import_jolpica.py` | `import-jolpica.yml` | ✅ Sostituito |
| `scripts/import_standings.py` | `import-jolpica.yml` | ✅ Sostituito |
| `scripts/import_tracinginsights.py` | `import-tracinginsights.yml` | ✅ In uso dal workflow |
| `scripts/import_kaggle.py` | Non implementato | ⚠️ Da rimuovere o implementare |
| `scripts/import_pinnacle_odds.py` | Escluso | ❌ Non rilevante |

## Documentazione Ridondante

| File | Note |
|------|------|
| `BetBreaker_F1_Documentazione_Completa.docx` | Doc vecchia, esiste già README.md |
| `F1_Predictor_Documentazione_Tecnica.docx` | Doc vecchia |
| `PROGRESS_SUMMARY.md` | Potrebbe essere ridondante con README |

## Test File

| File | Note |
|------|------|
| `tests/data/test_loader_jolpica.py` | Test per loader obsoleto - da aggiornare |

---

## Piano di Rimozione

### Fase 1: Rimozioni Immediate (Sicure)
```bash
# Cartelle vuote
rm -rf data/racedata/

# Script non più usati
rm scripts/import_kaggle.py  # Mai implementato completamente
```

### Fase 2: Dopo Verifica MongoDB
```bash
# Dopo conferma che tutti i dati sono in MongoDB
rm -rf data/cache/jolpica/
```

### Fase 3: Dopo Migrazione Completa
```bash
# Dopo verifica che i loader legacy non sono più necessari
rm f1_predictor/data/loader_jolpica.py
rm f1_predictor/data/loader_tracinginsights.py
rm f1_predictor/data/loader_kaggle.py
rm tests/data/test_loader_jolpica.py
```

## Nota Importante

**NON rimuovere** senza eseguire prima i regression test:
```bash
pytest tests/regression/ -v
```

Questo garantisce che la rimozione non rompe nulla.
