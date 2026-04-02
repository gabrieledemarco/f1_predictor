"""
f1_predictor/data/circuit_profiles.py
=======================================
Catalogo dei profili cinematici per tutti i circuiti del calendario 2026.

Fonti dati:
  - FastF1 telemetria (2023-2025 medie multi-stagione)
  - FIA Circuit Guides ufficiali
  - Heilmeier et al. (2020) Tab.1 — cluster caratteristici
  - Dati raccolti manualmente da motorsport-stats.com e f1-fansite.com

Classificazione curve (Heilmeier et al. 2020 §3.1):
  Lente  : apex speed < 120 km/h
  Medie  : 120 <= apex speed < 200 km/h
  Veloci : apex speed >= 200 km/h

Convenzione velocita:
  - top_speed_kmh:         velocita massima raggiunta nel giro (DRS zone)
  - min_speed_kmh:         velocita minima (punto di frenata massima)
  - avg_speed_kmh:         velocita media dell'intero giro (lap_length / lap_time)
  - avg_slow_corner_kmh:   media delle apex speed delle curve lente
  - avg_fast_corner_kmh:   media delle apex speed delle curve veloci
  - avg_medium_corner_kmh: media delle apex speed delle curve medie
  - full_throttle_pct:     % giro trascorso a throttle >= 95%

Nota 2026 - nuovi regolamenti:
  I valori di velocita sono derivati da dati 2023-2025 e corretti con un
  fattore +2% per tenere conto dell'aumento di potenza dei nuovi PU ibridi.
  Il full_throttle_pct rimane invariato (e' una caratteristica geometrica
  del circuito, non della potenza del motore).
"""

from f1_predictor.models.machine_pace import CircuitSpeedProfile
from f1_predictor.domain.entities import CircuitType


# ---------------------------------------------------------------------------
# Dizionario principale: circuit_ref -> CircuitSpeedProfile
# ---------------------------------------------------------------------------

CIRCUIT_PROFILES: dict[str, CircuitSpeedProfile] = {

    # ── AUSTRALIA (Melbourne) ───────────────────────────────────────
    "albert_park": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=312.0,
        min_speed_kmh=55.0,
        avg_speed_kmh=218.0,
        avg_slow_corner_kmh=75.0,
        avg_fast_corner_kmh=238.0,
        avg_medium_corner_kmh=152.0,
        full_throttle_pct=63.0,
        circuit_ref="albert_park",
        source="fastf1_avg_2023_2025",
    ),

    # ── CINA (Shanghai) ─────────────────────────────────────────────
    "shanghai": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_DOWNFORCE,
        top_speed_kmh=327.0,
        min_speed_kmh=58.0,
        avg_speed_kmh=210.0,
        avg_slow_corner_kmh=70.0,
        avg_fast_corner_kmh=245.0,
        avg_medium_corner_kmh=148.0,
        full_throttle_pct=55.0,
        circuit_ref="shanghai",
        source="fastf1_avg_2024",
    ),

    # ── GIAPPONE (Suzuka) ───────────────────────────────────────────
    "suzuka": CircuitSpeedProfile(
        circuit_type=CircuitType.MIXED,
        top_speed_kmh=318.0,
        min_speed_kmh=60.0,
        avg_speed_kmh=232.0,
        avg_slow_corner_kmh=80.0,
        avg_fast_corner_kmh=275.0,
        avg_medium_corner_kmh=162.0,
        full_throttle_pct=65.0,
        circuit_ref="suzuka",
        source="fastf1_avg_2023_2025",
    ),

    # ── MIAMI ───────────────────────────────────────────────────────
    "miami": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=320.0,
        min_speed_kmh=52.0,
        avg_speed_kmh=215.0,
        avg_slow_corner_kmh=68.0,
        avg_fast_corner_kmh=252.0,
        avg_medium_corner_kmh=158.0,
        full_throttle_pct=61.0,
        circuit_ref="miami",
        source="fastf1_avg_2023_2025",
    ),

    # ── CANADA (Montreal — Circuit Gilles Villeneuve) ───────────────
    "villeneuve": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=330.0,
        min_speed_kmh=48.0,
        avg_speed_kmh=213.0,
        avg_slow_corner_kmh=63.0,
        avg_fast_corner_kmh=258.0,
        avg_medium_corner_kmh=155.0,
        full_throttle_pct=68.0,
        circuit_ref="villeneuve",
        source="fastf1_avg_2023_2025",
    ),

    # ── MONACO ──────────────────────────────────────────────────────
    "monaco": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=295.0,
        min_speed_kmh=42.0,
        avg_speed_kmh=162.0,
        avg_slow_corner_kmh=58.0,
        avg_fast_corner_kmh=220.0,
        avg_medium_corner_kmh=130.0,
        full_throttle_pct=42.0,
        circuit_ref="monaco",
        source="fastf1_avg_2023_2025",
    ),

    # ── BARCELLONA-CATALUNYA ─────────────────────────────────────────
    "catalunya": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_DOWNFORCE,
        top_speed_kmh=316.0,
        min_speed_kmh=65.0,
        avg_speed_kmh=216.0,
        avg_slow_corner_kmh=82.0,
        avg_fast_corner_kmh=268.0,
        avg_medium_corner_kmh=158.0,
        full_throttle_pct=67.0,
        circuit_ref="catalunya",
        source="fastf1_avg_2023_2025",
    ),

    # ── AUSTRIA (Red Bull Ring) ──────────────────────────────────────
    "red_bull_ring": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_SPEED,
        top_speed_kmh=330.0,
        min_speed_kmh=78.0,
        avg_speed_kmh=238.0,
        avg_slow_corner_kmh=95.0,
        avg_fast_corner_kmh=290.0,
        avg_medium_corner_kmh=172.0,
        full_throttle_pct=72.0,
        circuit_ref="red_bull_ring",
        source="fastf1_avg_2023_2025",
    ),

    # ── GRAN BRETAGNA (Silverstone) ──────────────────────────────────
    "silverstone": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_SPEED,
        top_speed_kmh=322.0,
        min_speed_kmh=68.0,
        avg_speed_kmh=242.0,
        avg_slow_corner_kmh=85.0,
        avg_fast_corner_kmh=295.0,
        avg_medium_corner_kmh=178.0,
        full_throttle_pct=63.0,
        circuit_ref="silverstone",
        source="fastf1_avg_2023_2025",
    ),

    # ── BELGIO (Spa-Francorchamps) ───────────────────────────────────
    "spa": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_SPEED,
        top_speed_kmh=352.0,
        min_speed_kmh=68.0,
        avg_speed_kmh=235.0,
        avg_slow_corner_kmh=85.0,
        avg_fast_corner_kmh=305.0,
        avg_medium_corner_kmh=172.0,
        full_throttle_pct=68.0,
        circuit_ref="spa",
        source="fastf1_avg_2023_2025",
    ),

    # ── UNGHERIA (Hungaroring) ───────────────────────────────────────
    "hungaroring": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_DOWNFORCE,
        top_speed_kmh=312.0,
        min_speed_kmh=58.0,
        avg_speed_kmh=198.0,
        avg_slow_corner_kmh=72.0,
        avg_fast_corner_kmh=235.0,
        avg_medium_corner_kmh=145.0,
        full_throttle_pct=52.0,
        circuit_ref="hungaroring",
        source="fastf1_avg_2023_2025",
    ),

    # ── OLANDA (Zandvoort) ──────────────────────────────────────────
    "zandvoort": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_DOWNFORCE,
        top_speed_kmh=313.0,
        min_speed_kmh=62.0,
        avg_speed_kmh=215.0,
        avg_slow_corner_kmh=78.0,
        avg_fast_corner_kmh=248.0,
        avg_medium_corner_kmh=155.0,
        full_throttle_pct=59.0,
        circuit_ref="zandvoort",
        source="fastf1_avg_2023_2025",
    ),

    # ── ITALIA (Monza) ───────────────────────────────────────────────
    "monza": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_SPEED,
        top_speed_kmh=362.0,
        min_speed_kmh=62.0,
        avg_speed_kmh=254.0,
        avg_slow_corner_kmh=80.0,
        avg_fast_corner_kmh=310.0,
        avg_medium_corner_kmh=168.0,
        full_throttle_pct=79.0,
        circuit_ref="monza",
        source="fastf1_avg_2023_2025",
    ),

    # ── SPAGNA MADRID (Madring — nuovo 2026) ─────────────────────────
    "madrid": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=308.0,
        min_speed_kmh=50.0,
        avg_speed_kmh=205.0,
        avg_slow_corner_kmh=68.0,
        avg_fast_corner_kmh=238.0,
        avg_medium_corner_kmh=148.0,
        full_throttle_pct=58.0,
        circuit_ref="madrid",
        source="estimated_2026",   # circuito nuovo, dati stimati
    ),

    # ── AZERBAIJAN (Baku) ────────────────────────────────────────────
    "baku": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=370.0,
        min_speed_kmh=42.0,
        avg_speed_kmh=212.0,
        avg_slow_corner_kmh=55.0,
        avg_fast_corner_kmh=292.0,
        avg_medium_corner_kmh=152.0,
        full_throttle_pct=65.0,
        circuit_ref="baku",
        source="fastf1_avg_2023_2025",
    ),

    # ── SINGAPORE ───────────────────────────────────────────────────
    "marina_bay": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=302.0,
        min_speed_kmh=42.0,
        avg_speed_kmh=178.0,
        avg_slow_corner_kmh=60.0,
        avg_fast_corner_kmh=218.0,
        avg_medium_corner_kmh=135.0,
        full_throttle_pct=40.0,
        circuit_ref="marina_bay",
        source="fastf1_avg_2023_2025",
    ),

    # ── USA (Circuit of the Americas — Austin) ───────────────────────
    "americas": CircuitSpeedProfile(
        circuit_type=CircuitType.MIXED,
        top_speed_kmh=320.0,
        min_speed_kmh=58.0,
        avg_speed_kmh=212.0,
        avg_slow_corner_kmh=78.0,
        avg_fast_corner_kmh=260.0,
        avg_medium_corner_kmh=155.0,
        full_throttle_pct=62.0,
        circuit_ref="americas",
        source="fastf1_avg_2023_2025",
    ),

    # ── MESSICO (Autodromo Hermanos Rodriguez) ───────────────────────
    "rodriguez": CircuitSpeedProfile(
        circuit_type=CircuitType.MIXED,
        top_speed_kmh=368.0,
        min_speed_kmh=55.0,
        avg_speed_kmh=218.0,
        avg_slow_corner_kmh=72.0,
        avg_fast_corner_kmh=295.0,
        avg_medium_corner_kmh=162.0,
        full_throttle_pct=72.0,
        circuit_ref="rodriguez",
        source="fastf1_avg_2023_2025",
    ),

    # ── BRASILE (Interlagos) ─────────────────────────────────────────
    "interlagos": CircuitSpeedProfile(
        circuit_type=CircuitType.MIXED,
        top_speed_kmh=315.0,
        min_speed_kmh=72.0,
        avg_speed_kmh=215.0,
        avg_slow_corner_kmh=88.0,
        avg_fast_corner_kmh=265.0,
        avg_medium_corner_kmh=160.0,
        full_throttle_pct=65.0,
        circuit_ref="interlagos",
        source="fastf1_avg_2023_2025",
    ),

    # ── LAS VEGAS ────────────────────────────────────────────────────
    "las_vegas": CircuitSpeedProfile(
        circuit_type=CircuitType.STREET,
        top_speed_kmh=342.0,
        min_speed_kmh=52.0,
        avg_speed_kmh=225.0,
        avg_slow_corner_kmh=65.0,
        avg_fast_corner_kmh=278.0,
        avg_medium_corner_kmh=162.0,
        full_throttle_pct=72.0,
        circuit_ref="las_vegas",
        source="fastf1_2023_2024",
    ),

    # ── QATAR (Lusail) ───────────────────────────────────────────────
    "losail": CircuitSpeedProfile(
        circuit_type=CircuitType.HIGH_SPEED,
        top_speed_kmh=322.0,
        min_speed_kmh=82.0,
        avg_speed_kmh=238.0,
        avg_slow_corner_kmh=98.0,
        avg_fast_corner_kmh=288.0,
        avg_medium_corner_kmh=172.0,
        full_throttle_pct=70.0,
        circuit_ref="losail",
        source="fastf1_avg_2023_2024",
    ),

    # ── ABU DHABI (Yas Marina) ───────────────────────────────────────
    "yas_marina": CircuitSpeedProfile(
        circuit_type=CircuitType.DESERT,
        top_speed_kmh=332.0,
        min_speed_kmh=62.0,
        avg_speed_kmh=218.0,
        avg_slow_corner_kmh=78.0,
        avg_fast_corner_kmh=262.0,
        avg_medium_corner_kmh=158.0,
        full_throttle_pct=62.0,
        circuit_ref="yas_marina",
        source="fastf1_avg_2023_2025",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_profile(circuit_ref: str) -> CircuitSpeedProfile:
    """
    Restituisce il profilo per un circuito.
    Raises KeyError se il circuito non e' nel catalogo.
    """
    if circuit_ref not in CIRCUIT_PROFILES:
        raise KeyError(
            f"Circuito '{circuit_ref}' non trovato. "
            f"Disponibili: {sorted(CIRCUIT_PROFILES.keys())}"
        )
    return CIRCUIT_PROFILES[circuit_ref]


def get_profile_safe(
    circuit_ref: str,
    fallback_type: CircuitType = CircuitType.MIXED,
) -> CircuitSpeedProfile:
    """
    Versione safe: se il circuito non e' nel catalogo, restituisce
    un profilo generico con valori medi (usato come fallback in produzione).
    """
    if circuit_ref in CIRCUIT_PROFILES:
        return CIRCUIT_PROFILES[circuit_ref]

    # Profilo generico medio (usato se dati non disponibili)
    return CircuitSpeedProfile(
        circuit_type=fallback_type,
        top_speed_kmh=320.0,
        min_speed_kmh=62.0,
        avg_speed_kmh=218.0,
        avg_slow_corner_kmh=78.0,
        avg_fast_corner_kmh=260.0,
        avg_medium_corner_kmh=158.0,
        full_throttle_pct=62.0,
        circuit_ref=circuit_ref,
        source="generic_fallback",
    )


def list_circuits_by_type(circuit_type: CircuitType) -> list[str]:
    """Restituisce la lista di circuit_ref per un dato tipo."""
    return [
        ref for ref, profile in CIRCUIT_PROFILES.items()
        if profile.circuit_type == circuit_type
    ]
