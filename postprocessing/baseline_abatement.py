# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
EU abatement cost of each solved year against 2018.

EUR per tonne is (cost_year - cost_2018) / (CO2_2018 - CO2_year). Cost is
expanded capital plus operation, summed over the European Union and split
into the same technology groups as the Current Policies comparison. Existing
capital is not included, and the EU ETS price is not the cost. The emissions
change plotted on the left is CO2_year - CO2_2018. 2018 is the baseline, so
it is not a point on the figure.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from postprocessing.postprocess_iea_cps import _as_carrier_series, co2_mt

logger = logging.getLogger(__name__)

BASELINE_YEAR = 2018
ATMOSPHERE = "co2 atmosphere"
# Carrier of the atmosphere store. Its operating cost is the ETS price times
# tonnes, so it is left out of the project cost.
CARBON_PRICE_CARRIER = "co2"
META_COLUMNS = ("year", "eur_per_t", "emission_change", "cost_change")
REPORT_META = (
    "year",
    "eur_per_t",
    "emission_change_mt",
    "cost_change_eur",
    "co2_series",
)
EMISSION_COLOR = "#1f4e79"
# Same palette as the cost panel in the comparison figure, for any carrier
# that the technology map does not name.
FALLBACK_COLORS = [
    "#1f4e79",
    "#4c78a8",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#f58518",
    "#e45756",
    "#b279a2",
]

TECH_MAP = {
    "offwind-ac": "Offshore Wind",
    "offwind-dc": "Offshore Wind",
    "offwind-float": "Offshore Wind",
    "onwind": "Onshore Wind",
    "solar": "Solar PV",
    "solar rooftop": "Solar PV",
    "solar-hsat": "Solar PV",
    "csp": "Solar PV",
    "PHS": "Hydro",
    "hydro": "Hydro",
    "ror": "Hydro",
    "AC": "Grid Infrastructure",
    "DC": "Grid Infrastructure",
    "H2 pipeline": "Grid Infrastructure",
    "electricity distribution grid": "Grid Infrastructure",
    "gas pipeline": "Grid Infrastructure",
    "gas pipeline new": "Grid Infrastructure",
    "CO2 pipeline": "Grid Infrastructure",
    "rural air heat pump": "Heat Pumps",
    "rural ground heat pump": "Heat Pumps",
    "urban decentral air heat pump": "Heat Pumps",
    "urban central air heat pump": "Heat Pumps",
    "urban decentral water tanks discharger": "TES",
    "urban decentral water tanks charger": "TES",
    "urban decentral water tanks": "TES",
    "urban decentral heat vent": "TES",
    "rural heat vent": "TES",
    "urban central water tanks discharger": "TES",
    "urban central water tanks charger": "TES",
    "urban central water tanks": "TES",
    "urban central water pits discharger": "TES",
    "urban central water pits charger": "TES",
    "urban central water pits": "TES",
    "urban central heat vent": "TES",
    "rural water tanks discharger": "TES",
    "rural water tanks charger": "TES",
    "rural water tanks": "TES",
    "CCGT": "Gas-fired Powerplants",
    "OCGT": "Gas-fired Powerplants",
    "coal": "Coal",
    "lignite": "Coal",
    "coal for industry": "Coal",
    "H2 Electrolysis": "Hydrogen Production",
    "SMR": "Hydrogen Production",
    "SMR CC": "Hydrogen Production",
    "H2 Store": "Hydrogen Production",
    "Fischer-Tropsch": "Synthetic Fuels",
    "Haber-Bosch": "Synthetic Fuels",
    "Sabatier": "Synthetic Fuels",
    "biomass to liquid": "Synthetic Fuels",
    "biomass-to-methanol": "Synthetic Fuels",
    "methanolisation": "Synthetic Fuels",
    "rural biomass boiler": "Boilers",
    "rural gas boiler": "Boilers",
    "rural oil boiler": "Boilers",
    "urban central gas boiler": "Boilers",
    "urban central biomass boiler": "Boilers",
    "urban central oil boiler": "Boilers",
    "urban central coal boiler": "Boilers",
    "urban decentral biomass boiler": "Boilers",
    "urban decentral gas boiler": "Boilers",
    "urban decentral oil boiler": "Boilers",
    "urban decentral coal boiler": "Boilers",
    "rural coal boiler": "Boilers",
    "urban decentral solar thermal": "Solar Thermal",
    "urban central solar thermal": "Solar Thermal",
    "rural solar thermal": "Solar Thermal",
    "urban central solid biomass CHP CC": "CHP",
    "urban central solid biomass CHP": "CHP",
    "urban central gas CHP CC": "CHP",
    "urban central gas CHP": "CHP",
    "urban central coal CHP": "CHP",
    "urban decentral resistive heater": "Electric Heating",
    "urban central resistive heater": "Electric Heating",
    "rural resistive heater": "Electric Heating",
    "BEV charger": "Battery Storage",
    "EV battery": "Battery Storage",
    "battery": "Battery Storage",
    "battery charger": "Battery Storage",
    "battery discharger": "Battery Storage",
    "home battery": "Battery Storage",
    "home battery charger": "Battery Storage",
    "home battery discharger": "Battery Storage",
    "DAC": "CCS",
    "co2 sequestered": "CCS",
    "co2 stored": "CCS",
    "process emissions CC": "CCS",
    "process emissions": "CCS",
    "biogas": "Fuels",
    "biogas to gas": "Fuels",
    "unsustainable biogas": "Fuels",
    "unsustainable bioliquids": "Fuels",
    "gas": "Fuels",
    "gas for industry": "Fuels",
    "gas for industry CC": "Fuels",
    "non-sequestered HVC": "Fuels",
    "solid biomass": "Fuels",
    "solid biomass for industry": "Fuels",
    "solid biomass for industry CC": "Fuels",
    "unsustainable solid biomass": "Fuels",
    "oil": "Fuels",
    "oil primary": "Fuels",
    "oil refining": "Fuels",
    "land transport oil": "Fuels",
    "kerosene for aviation": "Fuels",
    "shipping oil": "Fuels",
    "agriculture machinery oil": "Fuels",
    "naphtha for industry": "Fuels",
    "uranium": "Nuclear",
    "ammonia cracker": "Synthetic Fuels",
    "ammonia store": "Synthetic Fuels",
    "methanol": "Synthetic Fuels",
    "industry methanol": "Synthetic Fuels",
    "HVC to air": "Fuels",
    "V2G": "Other Generation",
    "H2 Fuel Cell": "Other Generation",
    "H2 turbine": "Other Generation",
    "geothermal": "Other Generation",
    "marine": "Other Generation",
    "nuclear": "Nuclear",
    "load": "Other",
}

COLOR_MAP = {
    "Offshore Wind": "#6895dd",
    "Onshore Wind": "#235ebc",
    "Solar PV": "#f9d002",
    "Hydro": "#298c81",
    "Grid Infrastructure": "#110d63",
    "Heat Pumps": "#2fb537",
    "TES": "#d96f4c",
    "Gas-fired Powerplants": "#ff8c00",
    "Coal": "#545454",
    "Hydrogen Production": "#bf13a0",
    "Synthetic Fuels": "#46caf0",
    "Boilers": "#ffb07c",
    "Solar Thermal": "#fdb915",
    "CHP": "#8d5e56",
    "Electric Heating": "#d8f9b8",
    "Battery Storage": "#11875d",
    "CCS": "#ff5270",
    "Fuels": "#c9c9c9",
    "Other Generation": "#9a0200",
    "Nuclear": "#6b3fa0",
    "Other": "#888888",
}


def eur_per_tonne(
    cost_year: float,
    cost_baseline: float,
    co2_baseline: float,
    co2_year: float,
) -> float:
    """EUR per tonne from the cost and CO2 of one year against the baseline.

    The denominator is zero when the year emits the same CO2 as 2018. Callers
    skip that year instead of calling this.
    """
    denominator = co2_baseline - co2_year
    if denominator == 0:
        raise ZeroDivisionError(
            f"CO2 equals the {BASELINE_YEAR} baseline, so EUR/t is undefined."
        )
    return (cost_year - cost_baseline) / denominator


def group_carriers(costs: pd.Series) -> pd.Series:
    """Sum carrier costs into technology groups. Missing names stay as they are.

    The atmosphere-store carrier is the ETS charge and is removed before the
    sum, so the group total is the project cost.
    """
    if costs is None or len(costs) == 0:
        return pd.Series(dtype=float)
    costs = costs.copy()
    costs.index = costs.index.map(str)
    if CARBON_PRICE_CARRIER in costs.index:
        carbon = float(costs.loc[CARBON_PRICE_CARRIER].sum())
        if carbon != 0.0:
            logger.info(
                "Leaving %.6g EUR of atmosphere-store carbon charges out of the "
                "abatement cost. That term is the ETS price times tonnes.",
                carbon,
            )
        costs = costs.drop(CARBON_PRICE_CARRIER)
    if costs.empty:
        return pd.Series(dtype=float)
    grouped = costs.groupby(costs.index.map(lambda name: TECH_MAP.get(name, name))).sum()
    return grouped.astype(float)


def group_columns(frame: pd.DataFrame) -> list[str]:
    """Technology-group columns, in legend order."""
    skip = set(META_COLUMNS) | set(REPORT_META)
    present = [name for name in frame.columns if name not in skip]
    preferred = [name for name in COLOR_MAP if name in present]
    rest = sorted(name for name in present if name not in preferred)
    return preferred + rest


def build_project_table(
    costs_by_year: dict[int, pd.Series],
    co2_by_year: dict[int, float],
    baseline: int = BASELINE_YEAR,
) -> pd.DataFrame:
    """One row per plotted year.

    ``costs_by_year`` is absolute cost by technology group. ``co2_by_year``
    uses one unit for every year; the ratio divides by that unit. The baseline
    year is not a row. A later year with the same CO2 is left out.
    """
    costs_by_year = {int(year): series for year, series in costs_by_year.items()}
    co2_by_year = {int(year): float(value) for year, value in co2_by_year.items()}
    if baseline not in costs_by_year or baseline not in co2_by_year:
        raise ValueError(f"The {baseline} baseline network is missing.")
    wide = pd.DataFrame(costs_by_year).T.fillna(0.0).sort_index()
    wide.index = wide.index.astype(int)
    base_cost = wide.loc[baseline]
    base_co2 = co2_by_year[baseline]
    rows = []
    years = sorted(set(wide.index.tolist()) | set(co2_by_year))
    for year in years:
        if year == baseline:
            continue
        if year not in co2_by_year or year not in wide.index:
            logger.warning("Skipping %s because cost or CO2 is missing.", year)
            continue
        co2_year = co2_by_year[year]
        if co2_year == base_co2:
            logger.warning(
                "%s has the same CO2 as %s, so that year is not plotted.",
                year,
                baseline,
            )
            continue
        delta = wide.loc[year].astype(float) - base_cost.astype(float)
        numerator = float(delta.sum())
        ratio = eur_per_tonne(
            float(base_cost.sum()) + numerator,
            float(base_cost.sum()),
            base_co2,
            co2_year,
        )
        row = {
            "year": year,
            "eur_per_t": ratio,
            "emission_change": co2_year - base_co2,
            "cost_change": numerator,
        }
        row.update({str(name): float(value) for name, value in delta.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(META_COLUMNS))
    groups = group_columns(frame)
    keep = [name for name in groups if float(frame[name].abs().max()) > 1e-6]
    frame = frame[list(META_COLUMNS) + keep]
    # Groups that barely change are left out of the figure. The ratio is
    # rebuilt from the groups that remain, so the stack matches the x position.
    if keep:
        frame["cost_change"] = frame[keep].sum(axis=1)
    else:
        frame["cost_change"] = 0.0
    avoided = -frame["emission_change"]
    frame["eur_per_t"] = [
        eur_per_tonne(float(cost), 0.0, float(gap), 0.0)
        for cost, gap in zip(frame["cost_change"], avoided, strict=True)
    ]
    return frame.sort_values("year").reset_index(drop=True)


def _store_weights(n: pypsa.Network) -> pd.Series:
    weights = n.snapshot_weightings
    for column in ("stores", "objective", "generators"):
        if column in getattr(weights, "columns", []):
            return weights[column]
    return weights.iloc[:, 0]


def _dynamic_frame(n: pypsa.Network, attr: str) -> pd.DataFrame:
    container = getattr(n, "stores_t", None)
    if container is None:
        return pd.DataFrame()
    frame = getattr(container, attr, None)
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    return frame


def atmosphere_tonnes(n: pypsa.Network) -> float | None:
    """Annual net CO2 in the atmosphere store, in tonnes.

    The store level already includes snapshot weights. The last level minus
    the initial level is the modelled period; scaling by 8760 / weighted hours
    makes it one year. Returns None when the store, or its level, is absent.
    """
    stores = getattr(n, "stores", None)
    if stores is None or len(stores) == 0 or ATMOSPHERE not in stores.index:
        return None
    cyclic = False
    if "e_cyclic" in stores.columns and pd.notna(stores.at[ATMOSPHERE, "e_cyclic"]):
        cyclic = bool(stores.at[ATMOSPHERE, "e_cyclic"])
    period = None
    level = _dynamic_frame(n, "e")
    if not cyclic and ATMOSPHERE in level.columns and level[ATMOSPHERE].notna().any():
        series = level[ATMOSPHERE].dropna()
        initial = 0.0
        if "e_initial" in stores.columns and pd.notna(stores.at[ATMOSPHERE, "e_initial"]):
            initial = float(stores.at[ATMOSPHERE, "e_initial"])
        period = float(series.iloc[-1]) - initial
    if period is None:
        power = _dynamic_frame(n, "p")
        if ATMOSPHERE not in power.columns:
            return None
        # Store power is positive when the store discharges. Charging, which
        # is how emitted CO2 enters the store, is negative.
        weight = _store_weights(n).reindex(power.index).fillna(0.0)
        period = float((-power[ATMOSPHERE].fillna(0.0) * weight).sum())
    hours = float(_store_weights(n).sum())
    if hours <= 0:
        return float(period)
    return float(period) * 8760.0 / hours


def combustion_tonnes(n: pypsa.Network, year: int | None = None) -> float:
    """Annual coal, oil and gas combustion, in tonnes.

    This is the combustion total already reported by the comparison table.
    Process emissions are included in the fuel split when ``year`` is given.
    """
    emitted = co2_mt(n, year)
    return float(sum(emitted.get(fuel, 0.0) for fuel in ("coal", "oil", "gas"))) * 1e6


def objective_by_group(n: pypsa.Network) -> pd.Series:
    """Expanded capital plus operation by technology group, in EUR per year.

    ``nice_names`` stays off so carrier names match the technology map.
    Existing capital (installed capex) is not added.
    """
    stats = {"nice_names": False, "drop_zero": False, "round": 8}
    capital = _as_carrier_series(n.statistics.expanded_capex(**stats))
    operation = _as_carrier_series(n.statistics.opex(**stats))
    return group_carriers(capital.add(operation, fill_value=0.0))


def _year_of(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _read_network(path: Path) -> dict:
    year = _year_of(path)
    network = pypsa.Network(path)
    stored = atmosphere_tonnes(network)
    burned = combustion_tonnes(network, year)
    costs = objective_by_group(network)
    return {"year": year, "stored": stored, "combustion": burned, "costs": costs}


def _choose_co2(points: list[dict]) -> tuple[dict[int, float], str]:
    """One CO2 series for 2018 and every later year, in tonnes."""
    missing = [point["year"] for point in points if point["stored"] is None]
    if missing:
        logger.info(
            "CO2 atmosphere store is missing for %s. "
            "The combustion total is used for %s and every later year.",
            ", ".join(str(year) for year in sorted(missing)),
            BASELINE_YEAR,
        )
        return (
            {point["year"]: float(point["combustion"]) for point in points},
            "combustion",
        )
    logger.info(
        "CO2 is the co2 atmosphere store for %s and every later year.",
        BASELINE_YEAR,
    )
    return (
        {point["year"]: float(point["stored"]) for point in points},
        "co2 atmosphere",
    )


def report_table(table: pd.DataFrame, co2_series: str) -> pd.DataFrame:
    """Convert the working table to megatonnes and EUR for the csv."""
    if table.empty:
        columns = list(REPORT_META)
        return pd.DataFrame(columns=columns)
    out = table.copy()
    out["emission_change_mt"] = out["emission_change"] / 1e6
    out = out.rename(columns={"cost_change": "cost_change_eur"})
    out["co2_series"] = co2_series
    groups = group_columns(out)
    return out[list(REPORT_META) + groups]


def _cost_scale(frame: pd.DataFrame, groups: list[str]) -> tuple[float, str]:
    if not groups or frame.empty:
        return 1.0, "EUR"
    peak = float(frame[groups].abs().max().max())
    if peak >= 1e6:
        return 1e9, "billion EUR"
    return 1.0, "EUR"


def _bar_width(positions: np.ndarray) -> float:
    if len(positions) == 0:
        return 1.0
    if len(positions) == 1:
        return max(abs(float(positions[0])) * 0.12, 1.0)
    gaps = np.diff(np.sort(positions.astype(float)))
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return max(abs(float(positions[0])) * 0.12, 1.0)
    return float(np.min(gaps)) * 0.6


def plot_baseline_abatement(frame: pd.DataFrame, path: Path) -> list[int]:
    """Draw project years only. The baseline year is never a point.

    Returns the years that were drawn.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty or "year" not in frame.columns:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.set_xlabel("EUR/t = (cost − cost 2018) / (CO2 2018 − CO2)")
        ax.set_ylabel("Change in emissions versus 2018 [Mt]")
        ax.set_title(f"No project year. {BASELINE_YEAR} is the baseline.")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return []
    drawn = frame.loc[frame["year"].astype(int) != BASELINE_YEAR].copy()
    drawn["year"] = drawn["year"].astype(int)
    drawn = drawn.sort_values("year")
    if drawn.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.set_xlabel("EUR/t = (cost − cost 2018) / (CO2 2018 − CO2)")
        ax.set_ylabel("Change in emissions versus 2018 [Mt]")
        ax.set_title(f"No project year. {BASELINE_YEAR} is the baseline.")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return []
    groups = group_columns(drawn)
    total_column = "cost_change_eur" if "cost_change_eur" in drawn.columns else "cost_change"
    if groups and total_column in drawn.columns:
        got = drawn[groups].sum(axis=1).to_numpy(dtype=float)
        want = drawn[total_column].to_numpy(dtype=float)
        if not np.allclose(got, want, rtol=1e-6, atol=1e-4):
            raise ValueError(
                "Technology-group cost changes do not sum to the abatement numerator."
            )
    if "emission_change_mt" in drawn.columns:
        emissions = drawn["emission_change_mt"]
        emission_label = "Change in emissions versus 2018 [Mt]"
    else:
        emissions = drawn["emission_change"]
        emission_label = "Change in emissions versus 2018"
    scale, unit = _cost_scale(drawn, groups)
    positions = drawn["eur_per_t"].to_numpy(dtype=float)
    width = _bar_width(positions)
    fig, ax_emissions = plt.subplots(figsize=(11.2, 6.4))
    ax_cost = ax_emissions.twinx()
    handles = []
    labels = []
    seen = set()
    # Positive and negative changes each stack from zero, so a group that
    # costs less than in 2018 stays visible and the signed sum is the numerator.
    positive_bottom = np.zeros(len(drawn))
    negative_bottom = np.zeros(len(drawn))
    for index, name in enumerate(groups):
        values = drawn[name].to_numpy(dtype=float) / scale
        if np.all(np.abs(values) <= 1e-12):
            continue
        color = COLOR_MAP.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
        for row_index, value in enumerate(values):
            if value > 0:
                bottom = positive_bottom[row_index]
                positive_bottom[row_index] += value
            elif value < 0:
                bottom = negative_bottom[row_index]
                negative_bottom[row_index] += value
            else:
                continue
            bar = ax_cost.bar(
                positions[row_index],
                value,
                width=width,
                bottom=bottom,
                color=color,
                linewidth=0,
                zorder=2,
            )
            if name not in seen:
                handles.append(bar)
                labels.append(name)
                seen.add(name)
    ax_cost.axhline(0, color="#b0b0b0", linewidth=0.7, zorder=1)
    ax_cost.set_ylabel(f"Change in cost versus 2018 [{unit}]")
    line = ax_emissions.plot(
        positions,
        emissions,
        color=EMISSION_COLOR,
        marker="o",
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=1.1,
        zorder=5,
        label="Change in emissions versus 2018",
    )
    for year, x_value, y_value in zip(drawn["year"], positions, emissions, strict=True):
        ax_emissions.annotate(
            str(int(year)),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=EMISSION_COLOR,
            zorder=6,
        )
    handles = line + handles
    labels = ["Change in emissions versus 2018"] + labels
    ax_emissions.set_xlabel("EUR/t = (cost − cost 2018) / (CO2 2018 − CO2)")
    ax_emissions.set_ylabel(emission_label)
    low = min(0.0, float(emissions.min()))
    high = max(0.0, float(emissions.max()))
    pad = 0.08 * (high - low if high != low else 1.0)
    ax_emissions.set_ylim(low - pad, high + pad)
    ax_emissions.set_zorder(3)
    ax_cost.set_zorder(2)
    ax_emissions.patch.set_visible(False)
    for ax in (ax_emissions, ax_cost):
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)
        ax.set_axisbelow(True)
    ax_emissions.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(f"EU abatement versus {BASELINE_YEAR}")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [int(year) for year in drawn["year"].tolist()]


def write_baseline(networks: list[Path], output: Path) -> pd.DataFrame:
    """Read solved years and write the ratio table and the figure."""
    if not networks:
        raise ValueError("No solved networks were given.")
    points = [_read_network(path) for path in networks]
    by_year: dict[int, dict] = {}
    for point in points:
        by_year[point["year"]] = point
    co2, series = _choose_co2(list(by_year.values()))
    costs = {year: point["costs"] for year, point in by_year.items()}
    for year, grouped in costs.items():
        if float(grouped.sum()) == 0.0:
            logger.warning(
                "Expanded capital plus operation is zero for %s.",
                year,
            )
    logger.info(
        "EUR/t = (cost_year - cost_%s) / (CO2_%s - CO2_year), using %s.",
        BASELINE_YEAR,
        BASELINE_YEAR,
        series,
    )
    table = build_project_table(costs, co2, BASELINE_YEAR)
    report = report_table(table, series)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "baseline_abatement.csv"
    png_path = output / "baseline_abatement.png"
    report.to_csv(csv_path, index=False)
    plot_baseline_abatement(report, png_path)
    for row in report.itertuples(index=False):
        logger.info(
            "%s: %.2f EUR/t, emissions %+.3f Mt versus %s, cost %+.6g EUR versus %s.",
            int(row.year),
            float(row.eur_per_t),
            float(row.emission_change_mt),
            BASELINE_YEAR,
            float(row.cost_change_eur),
            BASELINE_YEAR,
        )
    if report.empty:
        logger.warning("No project year was plotted against %s.", BASELINE_YEAR)
    logger.info("Wrote %s", png_path)
    return report


def run_snakemake(snakemake) -> None:
    """Entry point used by the workflow after the solved networks exist."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    networks = snakemake.input.networks
    if isinstance(networks, str):
        networks = [networks]
    output = Path(snakemake.output.csv).parent
    write_baseline([Path(path) for path in networks], output)


def main() -> int:
    """Write the baseline abatement figure for the given solved networks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("networks", nargs="*", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/iea-cps/iea_comparison"),
    )
    args = parser.parse_args()
    networks = list(args.networks) or [
        Path(f"results/iea-cps/networks/solved_{year}.nc")
        for year in (2018, 2030, 2035, 2040)
    ]
    missing = [path for path in networks if not path.exists()]
    if missing:
        logger.error("Solved networks are not ready: %s", ", ".join(str(path) for path in missing))
        return 2
    try:
        write_baseline(networks, args.output)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    if "snakemake" in globals():
        run_snakemake(snakemake)  # noqa: F821
    else:
        raise SystemExit(main())
