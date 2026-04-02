"""
scripts/update_profiles.py
============================
CLI per aggiornare circuit_profiles.py con dati reali da FastF1.

Uso tipico (eseguire una volta per stagione, dopo le prime 3-4 gare):

    # Aggiornamento completo catalogo (anni 2022-2024)
    python scripts/update_profiles.py --years 2022 2023 2024

    # Solo circuiti specifici
    python scripts/update_profiles.py --years 2023 2024 --circuits monza suzuka spa

    # Solo qualifiche (più pulite della gara)
    python scripts/update_profiles.py --years 2024 --session Q

    # Dry run: stampa i risultati senza sovrascrivere il file
    python scripts/update_profiles.py --years 2024 --dry-run

    # Estrazione singolo circuito con report dettagliato
    python scripts/update_profiles.py --years 2023 2024 --circuits miami --verbose

Requisiti:
    pip install fastf1 scipy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup path per import relativi
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_profiles")


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggiorna circuit_profiles.py con dati reali da FastF1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--years", type=int, nargs="+", required=True,
        help="Anni da usare per l'estrazione (es. --years 2022 2023 2024)"
    )
    p.add_argument(
        "--circuits", type=str, nargs="+", default=None,
        help="Circuiti specifici da aggiornare (es. --circuits monza suzuka). "
             "Se omesso, aggiorna tutti i 22 del catalogo."
    )
    p.add_argument(
        "--session", type=str, choices=["Q", "R", "auto"], default="auto",
        help="Tipo di sessione: Q=qualifiche, R=gara, auto=prova Q poi R"
    )
    p.add_argument(
        "--cache-dir", type=str, default="data/cache/fastf1",
        help="Directory cache FastF1"
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path file di output. Default: f1_predictor/data/circuit_profiles.py"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Calcola i profili ma non sovrascrive il file"
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Mostra dettaglio dell'estrazione per ogni circuito"
    )
    p.add_argument(
        "--report", action="store_true",
        help="Stampa il report comparativo prima/dopo l'aggiornamento"
    )
    return p.parse_args()


def print_comparison(circuit_ref: str, before, after):
    """Stampa il confronto tra profilo manuale e estratto."""
    if before is None or after is None:
        return

    fields = [
        ("top_speed_kmh",        "Top speed (km/h)"),
        ("min_speed_kmh",        "Min speed (km/h)"),
        ("avg_speed_kmh",        "Avg speed (km/h)"),
        ("avg_slow_corner_kmh",  "Slow corners (km/h)"),
        ("avg_medium_corner_kmh","Medium corners (km/h)"),
        ("avg_fast_corner_kmh",  "Fast corners (km/h)"),
        ("full_throttle_pct",    "Full throttle (%)"),
    ]

    print(f"\n  {'─'*56}")
    print(f"  {circuit_ref.upper()}")
    print(f"  {'─'*56}")
    print(f"  {'Feature':<26} {'Manuale':>10} {'FastF1':>10} {'Delta':>8}")
    print(f"  {'─'*56}")
    for attr, label in fields:
        v_before = getattr(before, attr, None)
        v_after  = getattr(after, attr, None)
        if v_before is not None and v_after is not None:
            delta = v_after - v_before
            flag  = "  ←" if abs(delta) > 5 else ""
            print(f"  {label:<26} {v_before:>10.1f} {v_after:>10.1f} {delta:>+8.1f}{flag}")
    print(f"  Source: {after.source}")


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger("fastf1_extractor").setLevel(logging.DEBUG)

    # Import dopo path setup
    try:
        from f1_predictor.data.fastf1_extractor import FastF1CircuitExtractor
        from f1_predictor.data.circuit_profiles import CIRCUIT_PROFILES, get_profile_safe
    except ImportError as e:
        log.error(f"Import fallito: {e}")
        log.error("Assicurati di eseguire dallo stesso directory del repo.")
        sys.exit(1)

    extractor = FastF1CircuitExtractor(cache_dir=args.cache_dir)

    session_type = None if args.session == "auto" else args.session

    if args.report:
        # Modalità report: estrai e confronta senza aggiornare
        circuits = args.circuits or list(CIRCUIT_PROFILES.keys())
        print(f"\n{'='*60}")
        print(f"  REPORT CONFRONTO: manuale vs FastF1 ({args.years})")
        print(f"{'='*60}")

        for circuit_ref in circuits:
            before  = get_profile_safe(circuit_ref)
            after   = extractor.extract_profile(
                circuit_ref, args.years, session_type
            )
            print_comparison(circuit_ref, before, after)

        print()
        return

    # Aggiornamento catalogo
    status = extractor.update_catalog(
        years=args.years,
        circuits=args.circuits,
        output_path=args.output,
        dry_run=args.dry_run,
    )

    # Summary finale
    updated = [k for k, v in status.items() if v == "updated"]
    failed  = [k for k, v in status.items() if v == "failed"]

    print(f"\n{'='*50}")
    print(f"  RISULTATO AGGIORNAMENTO")
    print(f"{'='*50}")
    print(f"  Anni usati:    {args.years}")
    print(f"  Circuiti:      {len(status)} totali")
    print(f"  ✓ Aggiornati:  {len(updated)}")
    print(f"  ✗ Falliti:     {len(failed)}")

    if failed:
        print(f"\n  Falliti (mantenuto profilo manuale):")
        for c in failed:
            print(f"    - {c}")

    if args.dry_run:
        print(f"\n  DRY RUN — nessun file modificato")
    else:
        output = args.output or "f1_predictor/data/circuit_profiles.py"
        print(f"\n  File aggiornato: {output}")

    print()


if __name__ == "__main__":
    main()
