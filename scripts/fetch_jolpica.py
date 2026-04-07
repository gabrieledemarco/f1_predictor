"""
scripts/fetch_jolpica.py
========================
Scarica dati gara F1 da Jolpica API e li salva su MongoDB (jolpica_cache).

Usato da:
  - .github/workflows/fetch_data.yml  (automatico, GP weekend)
  - Manualmente per popolare MongoDB da zero

Uso:
    python scripts/fetch_jolpica.py --year 2026 --from-round 1 --to-round 5
    python scripts/fetch_jolpica.py --all-years                # 2018 → anno corrente
    python scripts/fetch_jolpica.py --year 2025 --dry-run      # stampa senza salvare

Requisiti:
    MONGODB_URI nel .env o come variabile d'ambiente
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Aggiungi la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now().year
FIRST_YEAR   = 2018


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch dati Jolpica → MongoDB jolpica_cache"
    )
    parser.add_argument("--year",       type=int, default=CURRENT_YEAR,
                        help=f"Anno da scaricare (default: {CURRENT_YEAR})")
    parser.add_argument("--from-round", type=int, default=1,
                        help="Primo round (default: 1)")
    parser.add_argument("--to-round",   type=int, default=24,
                        help="Ultimo round incluso (default: 24)")
    parser.add_argument("--all-years",  action="store_true",
                        help=f"Scarica tutti gli anni {FIRST_YEAR}→anno corrente")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Stampa senza salvare su MongoDB")
    args = parser.parse_args()

    # ── Connessione MongoDB ──────────────────────────────────────────────
    db = None
    if not args.dry_run:
        try:
            from core.db import get_db_direct, _resolve_uri
            db = get_db_direct()
            if db is None:
                if _resolve_uri():
                    log.error(
                        "URI MongoDB rilevata ma connessione fallita.\n"
                        "Cause tipiche: Atlas Network Access/IP allowlist, credenziali utente, DNS SRV.\n"
                        "Usa --dry-run per testare senza connessione."
                    )
                else:
                    log.error(
                        "MongoDB non configurato. Verifica MONGODB_URI/MONGO_URI nel .env o nei secrets.\n"
                        "Usa --dry-run per testare senza connessione."
                    )
                sys.exit(1)
            log.info(f"MongoDB connesso: {db.name}")
        except ImportError as e:
            log.error(f"Impossibile importare core.db: {e}")
            sys.exit(1)
    else:
        log.info("[DRY-RUN] Modalità dry-run — nessun dato verrà salvato")

    # ── Import loader ────────────────────────────────────────────────────
    try:
        from f1_predictor.data.loader_jolpica import JolpicaLoader
    except ImportError as e:
        log.error(f"Impossibile importare JolpicaLoader: {e}")
        sys.exit(1)

    # ── Determina anni da scaricare ──────────────────────────────────────
    if args.all_years:
        years = list(range(FIRST_YEAR, CURRENT_YEAR + 1))
    else:
        years = [args.year]

    log.info(
        f"Fetching anni: {years}  "
        f"| round: {args.from_round}–{args.to_round}"
        f"{'  | DRY-RUN' if args.dry_run else ''}"
    )

    # ── Fetch e salvataggio ──────────────────────────────────────────────
    total_inserted = 0
    total_errors   = 0

    loader = JolpicaLoader(
        cache_dir="data/cache/jolpica",
        force_refresh=False,
        db=db if not args.dry_run else None,
    )

    for year in years:
        log.info(f"\n── Anno {year} ──────────────────────────────────────")
        try:
            races = loader.load_season(year, through_round=args.to_round)

            # Filtra dal round minimo richiesto
            races = [r for r in races if r.get("round", 0) >= args.from_round]

            n = len(races)
            log.info(f"  {year}: {n} gare scaricate/aggiornate")
            total_inserted += n
        except Exception as exc:
            log.error(f"  {year}: errore — {exc}")
            total_errors += 1

    # ── Report finale ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Fetch completato {'(DRY-RUN)' if args.dry_run else ''}")
    print(f"  Anni processati : {len(years)}")
    print(f"  Gare elaborate  : {total_inserted}")
    print(f"  Errori          : {total_errors}")
    if not args.dry_run and db is not None:
        try:
            from core.db import jolpica_cache_collection
            n_total = jolpica_cache_collection(db).count_documents({})
            print(f"  jolpica_cache   : {n_total} documenti totali su MongoDB")
        except Exception:
            pass
    print("=" * 55 + "\n")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
