"""
data/fastf1_extractor.py
=========================
Estrae automaticamente CircuitSpeedProfile dalla telemetria FastF1.

Sostituisce i valori manuali in circuit_profiles.py con dati reali,
aggiornabili ogni stagione dopo che le prime gare sono state disputate.

Pipeline di estrazione per un circuito:
    1. Carica la sessione FastF1 (Qualifiche o Gara)
    2. Seleziona i giri rappresentativi (fastest lap per pilota,
       esclusi giri con SC/VSC, outlap, inlap)
    3. Estrae la telemetria aggregata (Speed, Throttle, X, Y)
    4. Calcola le 7 feature cinematiche:
         - top_speed_kmh       : max(Speed) sul giro rappresentativo
         - min_speed_kmh       : min(Speed) fuori dalla pit lane
         - avg_speed_kmh       : lap_distance / lap_time
         - avg_slow_corner_kmh : media apex speed curve < 120 km/h
         - avg_medium_corner_kmh: media apex speed curve 120-200 km/h
         - avg_fast_corner_kmh : media apex speed curve >= 200 km/h
         - full_throttle_pct   : % distanza con Throttle >= 95
    5. Media su N anni per stabilità (riduce varianza meteo/compound)
    6. Costruisce CircuitSpeedProfile con source="fastf1_auto"

Rilevamento corner (metodo apex-speed):
    Un corner è un minimo locale nella traccia Speed con:
      - profondità minima  : speed_drop >= MIN_CORNER_DROP (15 km/h)
      - distanza minima    : MIN_CORNER_DISTANCE (80 m tra apex successivi)
    Classificazione per apex speed (Heilmeier et al. 2020 §3.1):
      - Lente  : apex < SLOW_THRESHOLD  (120 km/h)
      - Medie  : SLOW_THRESHOLD <= apex < FAST_THRESHOLD (200 km/h)
      - Veloci : apex >= FAST_THRESHOLD (200 km/h)

Requisiti:
    pip install fastf1 scipy numpy

Cache:
    FastF1 usa la propria cache (default ~/.cache/fastf1 o la dir configurata).
    Per run ripetute il fetch è gratuito — i dati vengono letti da disco.

Riferimenti:
    - FastF1 docs: https://docs.fastf1.dev/
    - Heilmeier et al. (2020) §3.1 — classificazione corner per apex speed
    - arXiv 2512.00640 §2 — feature cinematiche F1
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger("fastf1_extractor")

# ---------------------------------------------------------------------------
# Costanti di tuning
# ---------------------------------------------------------------------------

# Corner detection
MIN_CORNER_DROP      = 15.0    # km/h: speed drop minimo per considerare un corner
MIN_CORNER_DISTANCE  = 80.0    # m: distanza minima tra apex consecutivi
SLOW_THRESHOLD       = 120.0   # km/h: soglia curve lente/medie
FAST_THRESHOLD       = 200.0   # km/h: soglia curve medie/veloci

# Selezione giri rappresentativi
THROTTLE_FULL        = 95.0    # soglia throttle "full" (%)
MIN_VALID_LAPS       = 5       # giri minimi per calcolare le statistiche
MIN_TYRE_LIFE        = 3       # esclude outlap (prime 3 tornate su gomme)

# Sessioni preferite (in ordine di priorità)
PREFERRED_SESSIONS   = ["Q", "R"]   # Qualifiche prima, poi Gara


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Risultato dell'estrazione per un circuito/anno.
    Contiene le feature grezze e i metadati di qualità.
    """
    circuit_ref:          str
    year:                 int
    session_type:         str      # "Q" | "R"

    # Feature estratte
    top_speed_kmh:        float
    min_speed_kmh:        float
    avg_speed_kmh:        float
    avg_slow_corner_kmh:  float
    avg_medium_corner_kmh: float
    avg_fast_corner_kmh:  float
    full_throttle_pct:    float

    # Qualità
    n_laps_used:          int
    n_slow_corners:       int
    n_medium_corners:     int
    n_fast_corners:       int
    warnings:             list[str] = field(default_factory=list)

    def is_reliable(self) -> bool:
        """True se l'estrazione ha abbastanza dati per essere affidabile."""
        return (
            self.n_laps_used >= MIN_VALID_LAPS
            and self.n_slow_corners + self.n_medium_corners + self.n_fast_corners >= 3
            and self.top_speed_kmh > 200
        )


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class FastF1CircuitExtractor:
    """
    Estrae CircuitSpeedProfile dalla telemetria FastF1.

    Usage:
        extractor = FastF1CircuitExtractor(cache_dir="data/cache/fastf1")

        # Singolo anno
        result = extractor.extract(circuit_ref="monza", year=2024)

        # Media multi-anno (consigliato per stabilità)
        profile = extractor.extract_profile(
            circuit_ref="monza",
            years=[2022, 2023, 2024],
        )

        # Aggiornamento del catalogo completo
        extractor.update_catalog(years=[2023, 2024])
    """

    def __init__(self, cache_dir: str = "data/cache/fastf1"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_fastf1()

    def _setup_fastf1(self):
        """Configura FastF1 con cache e logging minimale."""
        try:
            import fastf1
            fastf1.Cache.enable_cache(str(self.cache_dir))
            # Silenzia i log verbosi di FastF1
            logging.getLogger("fastf1").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            self._ff1 = fastf1
            log.info(f"FastF1 configurato — cache: {self.cache_dir}")
        except ImportError:
            raise ImportError(
                "FastF1 non installato. Esegui: pip install fastf1"
            )

    # ------------------------------------------------------------------
    # Estrazione singola sessione
    # ------------------------------------------------------------------

    def extract(
        self,
        circuit_ref: str,
        year: int,
        session_type: Optional[str] = None,
    ) -> Optional[ExtractionResult]:
        """
        Estrae le feature cinematiche per un circuito e anno.

        Args:
            circuit_ref: riferimento circuito (es. "monza", "miami")
            year:        anno della stagione (es. 2024)
            session_type: "Q" (qualifiche) o "R" (gara).
                          Se None, prova prima Q poi R.

        Returns:
            ExtractionResult oppure None se i dati non sono disponibili.
        """
        sessions_to_try = [session_type] if session_type else PREFERRED_SESSIONS

        for stype in sessions_to_try:
            try:
                result = self._extract_session(circuit_ref, year, stype)
                if result is not None:
                    return result
            except Exception as e:
                log.warning(f"  {circuit_ref} {year} {stype}: {e}")
                continue

        log.warning(f"Nessun dato disponibile per {circuit_ref} {year}")
        return None

    def _extract_session(
        self,
        circuit_ref: str,
        year: int,
        session_type: str,
    ) -> Optional[ExtractionResult]:
        """
        Carica una singola sessione FastF1 ed estrae le feature.
        """
        log.info(f"  Caricando {circuit_ref} {year} {session_type}...")

        # FastF1 usa il nome del circuito o il paese — mappa circuit_ref
        ff1_name = _CIRCUIT_REF_TO_FF1.get(circuit_ref, circuit_ref)

        session = self._ff1.get_session(year, ff1_name, session_type)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session.load(telemetry=True, laps=True, weather=False, messages=False)

        laps = self._select_representative_laps(session, session_type)

        if len(laps) < MIN_VALID_LAPS:
            log.warning(
                f"  {circuit_ref} {year} {session_type}: "
                f"solo {len(laps)} giri validi (min={MIN_VALID_LAPS})"
            )
            if len(laps) == 0:
                return None

        # Aggrega telemetria su tutti i giri selezionati
        all_speed    = []
        all_throttle = []
        all_distance = []
        lap_speeds   = []   # avg speed per giro (per calcolo media)
        extraction_warnings = []

        for _, lap in laps.iterrows():
            try:
                tel = lap.get_telemetry()
                if tel is None or len(tel) < 50:
                    continue

                speed    = tel["Speed"].values.astype(float)
                throttle = tel["Throttle"].values.astype(float)
                distance = tel["Distance"].values.astype(float)

                # Rimuovi punti con speed = 0 (pit lane / errori telemetria)
                valid = speed > 10
                speed    = speed[valid]
                throttle = throttle[valid]
                distance = distance[valid]

                if len(speed) < 50:
                    continue

                all_speed.append(speed)
                all_throttle.append(throttle)
                all_distance.append(distance)

                # Avg speed da lap_time e distanza
                lap_time_s = lap.LapTime.total_seconds() if hasattr(lap.LapTime, "total_seconds") else None
                lap_dist   = float(distance[-1] - distance[0]) if len(distance) > 1 else None
                if lap_time_s and lap_dist and lap_time_s > 30:
                    avg_s = (lap_dist / 1000.0) / (lap_time_s / 3600.0)
                    lap_speeds.append(avg_s)

            except Exception as e:
                extraction_warnings.append(f"lap skip: {e}")
                continue

        if not all_speed:
            return None

        # Concat tutti i giri per statistiche globali
        speed_concat    = np.concatenate(all_speed)
        throttle_concat = np.concatenate(all_throttle)

        # Feature scalari
        top_speed    = float(np.percentile(speed_concat, 99))   # 99° pct (evita spike)
        min_speed    = float(np.percentile(speed_concat, 1))    # 1° pct
        avg_speed    = float(np.mean(lap_speeds)) if lap_speeds else float(np.mean(speed_concat))
        full_thr_pct = float(np.mean(throttle_concat >= THROTTLE_FULL) * 100)

        # Corner detection su tutti i giri
        all_corner_apexes = []
        for speed_arr, dist_arr in zip(all_speed, all_distance):
            apexes = self._detect_corner_apexes(speed_arr, dist_arr)
            all_corner_apexes.extend(apexes)

        slow_apexes   = [v for v in all_corner_apexes if v < SLOW_THRESHOLD]
        medium_apexes = [v for v in all_corner_apexes if SLOW_THRESHOLD <= v < FAST_THRESHOLD]
        fast_apexes   = [v for v in all_corner_apexes if v >= FAST_THRESHOLD]

        avg_slow   = float(np.mean(slow_apexes))   if slow_apexes   else float(SLOW_THRESHOLD * 0.7)
        avg_medium = float(np.mean(medium_apexes)) if medium_apexes else float((SLOW_THRESHOLD + FAST_THRESHOLD) / 2)
        avg_fast   = float(np.mean(fast_apexes))   if fast_apexes   else float(FAST_THRESHOLD * 1.15)

        return ExtractionResult(
            circuit_ref=circuit_ref,
            year=year,
            session_type=session_type,
            top_speed_kmh=round(top_speed, 1),
            min_speed_kmh=round(min_speed, 1),
            avg_speed_kmh=round(avg_speed, 1),
            avg_slow_corner_kmh=round(avg_slow, 1),
            avg_medium_corner_kmh=round(avg_medium, 1),
            avg_fast_corner_kmh=round(avg_fast, 1),
            full_throttle_pct=round(full_thr_pct, 1),
            n_laps_used=len(all_speed),
            n_slow_corners=len(slow_apexes),
            n_medium_corners=len(medium_apexes),
            n_fast_corners=len(fast_apexes),
            warnings=extraction_warnings,
        )

    # ------------------------------------------------------------------
    # Selezione giri rappresentativi
    # ------------------------------------------------------------------

    def _select_representative_laps(self, session, session_type: str):
        """
        Seleziona i giri da usare per l'estrazione:
        - Qualifiche (Q): best lap di ogni pilota nel Q3/Q2
        - Gara (R):       giri puliti a mid-stint (esclusi sc/vsc/pit)

        Ritorna un DataFrame FastF1 di lap objects.
        """
        laps = session.laps

        if session_type == "Q":
            # Qualifiche: prendi il fastest lap per pilota
            # Filtra giri con tempo valido
            valid = laps[laps["LapTime"].notna()].copy()
            # Prendi il giro più veloce per pilota
            best = valid.loc[valid.groupby("Driver")["LapTime"].idxmin()]
            return best

        else:
            # Gara: prendi giri puliti (no pit, no sc, mid-stint)
            valid = laps[
                laps["LapTime"].notna() &
                laps["PitInTime"].isna() &
                laps["PitOutTime"].isna() &
                (laps["TyreLife"] >= MIN_TYRE_LIFE)
            ].copy()

            # Escludi prime 5 tornate (strategia, traffico)
            valid = valid[valid["LapNumber"] >= 5]

            # Prendi al massimo 8 giri per pilota per non sovrappesare
            valid = valid.groupby("Driver").apply(
                lambda g: g.nsmallest(8, "LapTime")
            ).reset_index(drop=True)

            return valid

    # ------------------------------------------------------------------
    # Corner detection
    # ------------------------------------------------------------------

    def _detect_corner_apexes(
        self,
        speed: np.ndarray,
        distance: np.ndarray,
    ) -> list[float]:
        """
        Rileva gli apex delle curve trovando i minimi locali nella
        traccia di velocità.

        Metodo:
          1. Smooth la traccia con una finestra mobile (riduce il rumore)
          2. Trova i minimi locali (scipy.signal.find_peaks su -speed)
          3. Filtra per profondità minima (MIN_CORNER_DROP)
          4. Filtra per distanza minima tra apexes (MIN_CORNER_DISTANCE)

        Returns:
            Lista delle apex speed in km/h per questo giro.
        """
        try:
            from scipy.signal import find_peaks, savgol_filter
        except ImportError:
            # Fallback senza scipy: usa rolling mean manuale
            return self._detect_corners_no_scipy(speed, distance)

        if len(speed) < 20:
            return []

        # Smooth per ridurre spike da telemetria
        window = min(11, len(speed) // 10 * 2 + 1)
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1

        try:
            speed_smooth = savgol_filter(speed, window_length=window, polyorder=2)
        except Exception:
            speed_smooth = speed.copy()

        # Distanza in metri tra campioni telemetria
        if len(distance) > 1:
            dist_per_sample = np.median(np.diff(distance))
            min_samples = max(1, int(MIN_CORNER_DISTANCE / dist_per_sample))
        else:
            min_samples = 10

        # Trova minimi locali (= inversione del segnale per find_peaks)
        peaks, properties = find_peaks(
            -speed_smooth,
            prominence=MIN_CORNER_DROP,
            distance=min_samples,
        )

        if len(peaks) == 0:
            return []

        return [float(speed_smooth[p]) for p in peaks]

    def _detect_corners_no_scipy(
        self,
        speed: np.ndarray,
        distance: np.ndarray,
    ) -> list[float]:
        """Fallback corner detection senza scipy."""
        apexes = []
        window = 10
        for i in range(window, len(speed) - window):
            local_min = speed[i]
            surround  = np.concatenate([speed[i-window:i], speed[i+1:i+window+1]])
            if local_min < surround.min() - MIN_CORNER_DROP / 2:
                # Controlla distanza dall'ultimo apex
                if apexes:
                    last_dist = distance[i] - distance[-1] if len(distance) > i else 0
                    if last_dist < MIN_CORNER_DISTANCE:
                        continue
                apexes.append(float(local_min))
        return apexes

    # ------------------------------------------------------------------
    # Aggregazione multi-anno
    # ------------------------------------------------------------------

    def extract_profile(
        self,
        circuit_ref: str,
        years: list[int],
        session_type: Optional[str] = None,
        min_years: int = 1,
    ):
        """
        Estrae e media i profili su più anni per stabilità.

        Args:
            circuit_ref:  riferimento circuito
            years:        lista di anni da usare (es. [2022, 2023, 2024])
            session_type: "Q", "R" o None (auto)
            min_years:    anni minimi per restituire un profilo

        Returns:
            CircuitSpeedProfile oppure None se dati insufficienti.
        """
        from f1_predictor.data.circuit_profiles import get_profile_safe
        from f1_predictor.domain.entities import CircuitType

        results = []
        for year in years:
            r = self.extract(circuit_ref, year, session_type)
            if r is not None and r.is_reliable():
                results.append(r)
            elif r is not None:
                log.warning(
                    f"  {circuit_ref} {year}: dati estratti ma non affidabili "
                    f"(n_laps={r.n_laps_used}, corners={r.n_slow_corners + r.n_medium_corners + r.n_fast_corners})"
                )

        if len(results) < min_years:
            log.warning(
                f"  {circuit_ref}: solo {len(results)}/{len(years)} anni affidabili — "
                f"uso profilo manuale come fallback"
            )
            return None

        # Media pesata per n_laps_used (anni con più giri pesano di più)
        weights = np.array([r.n_laps_used for r in results], dtype=float)
        weights /= weights.sum()

        def wavg(attr: str) -> float:
            return float(np.sum([getattr(r, attr) * w for r, w in zip(results, weights)]))

        # Determina il circuit_type dal profilo manuale esistente
        # (il tipo di circuito non cambia anno per anno)
        existing = get_profile_safe(circuit_ref)
        circuit_type = existing.circuit_type

        from f1_predictor.models.machine_pace import CircuitSpeedProfile

        profile = CircuitSpeedProfile(
            circuit_type=circuit_type,
            top_speed_kmh=round(wavg("top_speed_kmh"), 1),
            min_speed_kmh=round(wavg("min_speed_kmh"), 1),
            avg_speed_kmh=round(wavg("avg_speed_kmh"), 1),
            avg_slow_corner_kmh=round(wavg("avg_slow_corner_kmh"), 1),
            avg_medium_corner_kmh=round(wavg("avg_medium_corner_kmh"), 1),
            avg_fast_corner_kmh=round(wavg("avg_fast_corner_kmh"), 1),
            full_throttle_pct=round(wavg("full_throttle_pct"), 1),
            circuit_ref=circuit_ref,
            source=f"fastf1_auto_{min(r.year for r in results)}_{max(r.year for r in results)}",
        )

        log.info(
            f"  {circuit_ref}: profilo estratto da {len(results)} anni "
            f"(top={profile.top_speed_kmh}, throttle={profile.full_throttle_pct}%)"
        )
        return profile

    # ------------------------------------------------------------------
    # Aggiornamento catalogo
    # ------------------------------------------------------------------

    def update_catalog(
        self,
        years: list[int],
        circuits: Optional[list[str]] = None,
        output_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Aggiorna circuit_profiles.py con i dati estratti da FastF1.

        Args:
            years:       anni da usare per l'estrazione (es. [2022,2023,2024])
            circuits:    lista di circuit_ref da aggiornare.
                         Se None, aggiorna tutti i 22 del catalogo.
            output_path: path del file da sovrascrivere.
                         Default: f1_predictor/data/circuit_profiles.py
            dry_run:     se True, stampa senza scrivere su disco.

        Returns:
            Dict {circuit_ref: "updated"|"failed"|"skipped"} con lo stato
            di ogni circuito.
        """
        from f1_predictor.data.circuit_profiles import CIRCUIT_PROFILES

        target_circuits = circuits or list(CIRCUIT_PROFILES.keys())
        updated_profiles = dict(CIRCUIT_PROFILES)  # copia
        status = {}

        log.info(f"Aggiornamento catalogo: {len(target_circuits)} circuiti, anni {years}")

        for circuit_ref in target_circuits:
            log.info(f"[{circuit_ref}]")
            profile = self.extract_profile(circuit_ref, years)

            if profile is not None:
                updated_profiles[circuit_ref] = profile
                status[circuit_ref] = "updated"
                log.info(f"  ✓ aggiornato")
            else:
                status[circuit_ref] = "failed"
                log.warning(f"  ✗ fallito — mantenuto profilo manuale")

        if not dry_run:
            path = Path(output_path) if output_path else (
                Path(__file__).parent / "circuit_profiles.py"
            )
            self._write_catalog(updated_profiles, path, years)
            log.info(f"Catalogo scritto su: {path}")
        else:
            log.info("DRY RUN — nessun file scritto")

        n_ok   = sum(1 for v in status.values() if v == "updated")
        n_fail = sum(1 for v in status.values() if v == "failed")
        log.info(f"Risultato: {n_ok} aggiornati, {n_fail} falliti")

        return status

    def _write_catalog(
        self,
        profiles: dict,
        path: Path,
        years: list[int],
    ):
        """
        Riscrive circuit_profiles.py con i nuovi valori.
        Mantiene la struttura del file originale.
        """
        from f1_predictor.domain.entities import CircuitType

        lines = [
            '"""',
            "f1_predictor/data/circuit_profiles.py",
            "=" * 43,
            f"AUTO-GENERATED — last update: FastF1 anni {min(years)}-{max(years)}",
            "Non modificare manualmente i valori estratti (source='fastf1_auto_*').",
            "Per aggiornare: python -m f1_predictor.data.update_profiles --years 2024 2025",
            '"""',
            "",
            "from f1_predictor.models.machine_pace import CircuitSpeedProfile",
            "from f1_predictor.domain.entities import CircuitType",
            "",
            "",
            "CIRCUIT_PROFILES: dict[str, CircuitSpeedProfile] = {",
            "",
        ]

        ct_map = {
            "street":       "CircuitType.STREET",
            "high_df":      "CircuitType.HIGH_DOWNFORCE",
            "high_speed":   "CircuitType.HIGH_SPEED",
            "mixed":        "CircuitType.MIXED",
            "desert":       "CircuitType.DESERT",
        }

        for ref, p in profiles.items():
            ct_str = ct_map.get(p.circuit_type.value, "CircuitType.MIXED")
            lines += [
                f'    "{ref}": CircuitSpeedProfile(',
                f"        circuit_type={ct_str},",
                f"        top_speed_kmh={p.top_speed_kmh},",
                f"        min_speed_kmh={p.min_speed_kmh},",
                f"        avg_speed_kmh={p.avg_speed_kmh},",
                f"        avg_slow_corner_kmh={p.avg_slow_corner_kmh},",
                f"        avg_fast_corner_kmh={p.avg_fast_corner_kmh},",
                f"        avg_medium_corner_kmh={p.avg_medium_corner_kmh},",
                f"        full_throttle_pct={p.full_throttle_pct},",
                f'        circuit_ref="{ref}",',
                f'        source="{p.source}",',
                f"    ),",
                "",
            ]

        lines += [
            "}",
            "",
            "",
            "# ---------------------------------------------------------------------------",
            "# Lookup helpers (invariati)",
            "# ---------------------------------------------------------------------------",
            "",
            "def get_profile(circuit_ref: str) -> CircuitSpeedProfile:",
            '    if circuit_ref not in CIRCUIT_PROFILES:',
            '        raise KeyError(f"Circuito \'{circuit_ref}\' non trovato. '
            'Disponibili: {sorted(CIRCUIT_PROFILES.keys())}")',
            "    return CIRCUIT_PROFILES[circuit_ref]",
            "",
            "",
            "def get_profile_safe(",
            "    circuit_ref: str,",
            "    fallback_type=None,",
            ") -> CircuitSpeedProfile:",
            "    from f1_predictor.domain.entities import CircuitType",
            "    if circuit_ref in CIRCUIT_PROFILES:",
            "        return CIRCUIT_PROFILES[circuit_ref]",
            "    ft = fallback_type or CircuitType.MIXED",
            "    return CircuitSpeedProfile(",
            "        circuit_type=ft,",
            "        top_speed_kmh=320.0, min_speed_kmh=62.0, avg_speed_kmh=218.0,",
            "        avg_slow_corner_kmh=78.0, avg_fast_corner_kmh=260.0,",
            "        avg_medium_corner_kmh=158.0, full_throttle_pct=62.0,",
            '        circuit_ref=circuit_ref, source="generic_fallback",',
            "    )",
            "",
            "",
            "def list_circuits_by_type(circuit_type) -> list[str]:",
            "    return [ref for ref, p in CIRCUIT_PROFILES.items()",
            "            if p.circuit_type == circuit_type]",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Mappa circuit_ref → nome FastF1
# ---------------------------------------------------------------------------
# FastF1 usa il nome del GP o il paese per identificare il circuito.
# Questa mappa converte i nostri circuit_ref in nomi che FastF1 accetta.

_CIRCUIT_REF_TO_FF1: dict[str, str] = {
    "albert_park":  "Australia",
    "shanghai":     "China",
    "suzuka":       "Japan",
    "miami":        "Miami",
    "villeneuve":   "Canada",
    "monaco":       "Monaco",
    "catalunya":    "Spain",          # Barcelona-Catalunya
    "red_bull_ring": "Austria",
    "silverstone":  "Great Britain",
    "spa":          "Belgium",
    "hungaroring":  "Hungary",
    "zandvoort":    "Netherlands",
    "monza":        "Italy",
    "madrid":       "Spain",          # nuovo 2026 — non disponibile pre-2026
    "baku":         "Azerbaijan",
    "marina_bay":   "Singapore",
    "americas":     "USA",
    "rodriguez":    "Mexico",
    "interlagos":   "Brazil",
    "las_vegas":    "Las Vegas",
    "losail":       "Qatar",
    "yas_marina":   "Abu Dhabi",
}
