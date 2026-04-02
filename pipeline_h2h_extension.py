"""
Estensione di F1PredictionPipeline per supporto H2H.

Aggiungere questo metodo alla classe F1PredictionPipeline in pipeline.py.

Oppure usarlo come mixin separato:
    class F1PredictionPipelineH2HMixin:
        def predict_h2h(self, ...): ...

Poi nella classe principale:
    class F1PredictionPipeline(F1PredictionPipelineH2HMixin):
        ...
"""
from __future__ import annotations

from typing import Optional


# ============================================================
# Incolla questo metodo nella classe F1PredictionPipeline
# in f1_predictor/pipeline.py
# ============================================================

PIPELINE_H2H_METHOD = '''
    def predict_h2h(
        self,
        race_probs: dict,
        driver_to_constructor: dict,
        driver_odds: Optional[dict] = None,
        constructor_odds: Optional[dict] = None,
        teammate_only: bool = False,
        driver_pairs: Optional[list] = None,
        constructor_pairs: Optional[list] = None,
        race_name: str = "",
        min_edge: float = 0.04,
        sim_matrix: Optional["np.ndarray"] = None,
        driver_index: Optional[dict] = None,
    ) -> dict:
        """
        Calcola probabilità H2H driver e constructor, con edge opzionale vs bookmaker.

        Args:
            race_probs:           output di predict_race()["probabilities"]
            driver_to_constructor: {driver_code: constructor_ref}
                                   es. {"VER": "red_bull", "PER": "red_bull", ...}
            driver_odds:          {(da, db): {da: odd, db: odd}}
                                   es. {("VER","NOR"): {"VER": 1.65, "NOR": 2.20}}
            constructor_odds:     {(ta, tb): {ta: odd, tb: odd}}
            teammate_only:        se True calcola H2H solo tra compagni di squadra
            driver_pairs:         lista coppie driver specifiche (override teammate_only)
            constructor_pairs:    lista coppie team specifiche
            race_name:            nome GP per il report
            min_edge:             soglia minima per value bets
            sim_matrix:           matrice MC raw opzionale (n_sims × n_drivers)
            driver_index:         {driver_code: column_index} per sim_matrix

        Returns:
            {
                "driver_h2h":        list[DriverH2H],
                "constructor_h2h":   list[ConstructorH2H],
                "edge_report":       H2HEdgeReport | None,
                "teammate_h2h":      list[DriverH2H],    # sempre calcolato
            }
        """
        from f1_predictor.h2h import H2HCalculator, H2HEdgeCalculator

        calc = H2HCalculator()
        kwargs = dict(
            sim_matrix=sim_matrix,
            driver_index=driver_index,
        )

        # ── Driver H2H ──────────────────────────────────────────────
        if driver_pairs is not None:
            driver_h2h = calc.all_driver_h2h(
                race_probs, pairs=driver_pairs, **kwargs
            )
        elif teammate_only:
            driver_h2h = calc.teammate_pairs(
                race_probs, driver_to_constructor, **kwargs
            )
        else:
            driver_h2h = calc.all_driver_h2h(race_probs, **kwargs)

        # Compagni di squadra sempre calcolati (utile per BetBreaker)
        teammate_h2h = calc.teammate_pairs(
            race_probs, driver_to_constructor, **kwargs
        )

        # ── Constructor H2H ─────────────────────────────────────────
        constructor_h2h = calc.all_constructor_h2h(
            race_probs, driver_to_constructor,
            pairs=constructor_pairs, **kwargs
        )

        # ── Edge report ─────────────────────────────────────────────
        edge_report = None
        if driver_odds or constructor_odds:
            edge_calc = H2HEdgeCalculator(
                min_edge=min_edge,
                kelly_fraction=0.25,
                devig_method="power",
            )
            edge_report = edge_calc.build_report(
                race_name=race_name,
                driver_h2h_list=driver_h2h,
                constructor_h2h_list=constructor_h2h,
                driver_odds=driver_odds or {},
                constructor_odds=constructor_odds or {},
            )

        return {
            "driver_h2h": driver_h2h,
            "constructor_h2h": constructor_h2h,
            "teammate_h2h": teammate_h2h,
            "edge_report": edge_report,
        }
'''

# ============================================================
# Esempio d'uso completo
# ============================================================

USAGE_EXAMPLE = '''
# ── Esempio d'uso — GP Miami 2026 ──────────────────────────────────────

from f1_predictor.pipeline import F1PredictionPipeline

pipeline = F1PredictionPipeline()
pipeline.fit(historical_races)

# 1. Predici la gara (come sempre)
race_result = pipeline.predict_race(
    race=miami_race,
    driver_grid=driver_grid,
    pinnacle_odds={"VER": 2.10, "NOR": 4.50, ...}
)
race_probs = race_result["probabilities"]

# 2. Mappa pilota → constructor
driver_to_constructor = {
    "VER": "red_bull", "TSU": "red_bull",
    "NOR": "mclaren",  "PIA": "mclaren",
    "LEC": "ferrari",  "HAM": "ferrari",
    "RUS": "mercedes", "ANT": "mercedes",
    "ALO": "aston_martin", "STR": "aston_martin",
}

# 3. Quote bookmaker H2H (es. da Eurobet)
driver_odds = {
    ("VER", "NOR"):  {"VER": 1.65, "NOR": 2.20},
    ("LEC", "HAM"):  {"LEC": 1.85, "HAM": 1.95},
    ("RUS", "ANT"):  {"RUS": 1.90, "ANT": 1.90},
    ("NOR", "LEC"):  {"NOR": 1.75, "LEC": 2.05},
}

constructor_odds = {
    ("red_bull", "mclaren"):  {"red_bull": 1.55, "mclaren": 2.40},
    ("ferrari",  "mercedes"): {"ferrari": 1.70,  "mercedes": 2.10},
}

# 4. Calcola H2H + edge
h2h_result = pipeline.predict_h2h(
    race_probs=race_probs,
    driver_to_constructor=driver_to_constructor,
    driver_odds=driver_odds,
    constructor_odds=constructor_odds,
    race_name="Miami GP 2026",
    min_edge=0.04,
)

# 5. Stampa report
print(h2h_result["edge_report"].to_text())

# 6. Solo i compagni di squadra
for h2h in h2h_result["teammate_h2h"]:
    print(f"{h2h.driver_a} vs {h2h.driver_b}: "
          f"P({h2h driver_a})={h2h.p_a_beats_b:.3f}")

# 7. Value bets pronti per BetBreaker
value_bets = h2h_result["edge_report"].value_bets
for bet in value_bets:
    print(bet.to_text())

# Output esempio:
# ✅ VER vs NOR → VER           P_mod=0.682  P_fair=0.621  Odd=1.65  Edge=+0.061  EV=+10.0%  Kelly¼=2.3%
# ✅ red_bull vs mclaren → red_bull  P_mod=0.710  P_fair=0.649  Odd=1.55  Edge=+0.061  EV=+9.5%  Kelly¼=2.1%
# ⚠️ LEC vs HAM → LEC           P_mod=0.523  P_fair=0.497  Odd=1.85  Edge=+0.026  EV=+4.8%  Kelly¼=0.9%
'''
