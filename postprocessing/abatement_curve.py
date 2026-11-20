# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
EU abatement curve for the three Current Policies solves.

The country curves in the other study plot the CO2 price against the CO2
quantity, with system cost by technology stacked on an inverted axis. This
does the same for the European Union as one region. The price is the EU ETS
price charged on the atmosphere store. These networks do not keep the shadow
price of the combustion equalities.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

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
    "co2": "CO2 price",
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
    "CO2 price": "#6b3fa0",
    "Other Generation": "#9a0200",
    "Nuclear": "purple",
    "Other": "#888888",
}


def _by_carrier(value: pd.Series) -> pd.Series:
    """Sum a cost statistic down to one number per carrier."""
    if isinstance(value, pd.DataFrame):
        value = value.sum(axis=1)
    if not isinstance(value, pd.Series) or value.empty:
        return pd.Series(dtype=float)
    if isinstance(value.index, pd.MultiIndex):
        names = list(value.index.names)
        level = "carrier" if "carrier" in names else names[-1]
        value = value.groupby(level=level).sum()
    value.index = value.index.map(str)
    return value.groupby(level=0).sum()


def technology_costs(n: pypsa.Network) -> pd.Series:
    """Capital plus operation by technology group, in billion euro."""
    capital = _by_carrier(n.statistics.capex(nice_names=False))
    operation = _by_carrier(n.statistics.opex(nice_names=False))
    combined = capital.add(operation, fill_value=0.0)
    grouped = combined.groupby(combined.index.map(lambda name: TECH_MAP.get(name, name))).sum()
    return grouped / 1e9


def co2_point(n: pypsa.Network, year: int) -> dict:
    """ETS price and the CO2 that reached the atmosphere."""
    price = abs(float(n.stores.at["co2 atmosphere", "marginal_cost"]))
    emitted = float(n.stores_t.e["co2 atmosphere"].iloc[-1]) / 1e6
    return {
        "year": year,
        "co2_price": price,
        "co2_mt": emitted,
        "objective_bn": float(n.objective) / 1e9,
    }


def combustion_mt(comparison: Path) -> pd.Series:
    """Published-comparison combustion total, if that table already exists."""
    if not comparison.exists():
        return pd.Series(dtype=float)
    table = pd.read_csv(comparison)
    rows = table.loc[table.category.eq("co2") & table.name.eq("total")]
    return rows.set_index("year")["model"].astype(float)


def plot_curve(summary: pd.DataFrame, costs: pd.DataFrame, path: Path) -> None:
    """One EU panel: price on x, emissions as a line, costs stacked and inverted."""
    summary = summary.sort_values("co2_price")
    prices = summary["co2_price"].tolist()
    pivot = costs.pivot_table(index="co2_price", columns="tech", values="costs", aggfunc="sum").fillna(0.0)
    pivot = pivot.reindex(prices).fillna(0.0)
    small = [
        name
        for name in pivot.columns
        if name != "Other" and pivot[name].abs().max() < 0.5
    ]
    if small:
        extra = pivot[small].sum(axis=1)
        if "Other" in pivot:
            pivot["Other"] = pivot["Other"].add(extra, fill_value=0.0)
        else:
            pivot["Other"] = extra
        pivot = pivot.drop(columns=small)
    order = pivot.sum().sort_values(ascending=False).index
    pivot = pivot[order]

    fig, ax1 = plt.subplots(figsize=(10.5, 6.4))
    ax1.set_xlabel("CO2 price [€/tCO2]")
    ax1.set_ylabel("CO2 [Mt]")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2 = ax1.twinx()
    ax2.stackplot(
        prices,
        pivot.T.to_numpy(),
        labels=list(pivot.columns),
        colors=[COLOR_MAP.get(tech, "#333333") for tech in pivot.columns],
    )
    ax2.set_ylabel("System costs [billion euro]")
    ax2.invert_yaxis()
    ax2.legend(loc="upper left", bbox_to_anchor=(1.08, 1), fontsize=8, frameon=False)
    # The cost axes sits on top of the emissions axes. Lift the line back above it.
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    low = float(summary["co2_mt"].min())
    if "combustion_mt" in summary and summary["combustion_mt"].notna().any():
        low = min(low, float(summary["combustion_mt"].min()))
    high = float(summary["co2_mt"].max())
    pad = 0.08 * (high - low)
    ax1.set_ylim(low - pad, high + pad)
    ax1.plot(
        summary["co2_price"],
        summary["co2_mt"],
        color="#1f4e79",
        marker="o",
        linewidth=2.4,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=5,
        label="Atmosphere",
    )
    if "combustion_mt" in summary and summary["combustion_mt"].notna().any():
        ax1.plot(
            summary["co2_price"],
            summary["combustion_mt"],
            color="#c47b2b",
            marker="s",
            linewidth=2.0,
            linestyle="--",
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=5,
            label="Combustion",
        )
    for row in summary.itertuples(index=False):
        ax1.annotate(
            str(int(row.year)),
            (row.co2_price, row.co2_mt),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=8,
            color="#1f4e79",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    ax1.legend(loc="upper left", bbox_to_anchor=(1.08, 0.12), fontsize=8, frameon=False)
    ax1.set_xlim(min(prices) - 1.5, max(prices) + 2.5)
    fig.suptitle("EU. CO2 price is the ETS price charged on the atmosphere.")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_curve(networks: list[Path], output: Path, comparison: Path | None = None) -> None:
    """Write the EU price, emissions and cost curve for the solved years."""
    output.mkdir(parents=True, exist_ok=True)
    points = []
    cost_rows = []
    for path in networks:
        year = int(path.stem.split("_")[-1])
        network = pypsa.Network(path)
        point = co2_point(network, year)
        points.append(point)
        for tech, value in technology_costs(network).items():
            cost_rows.append({
                "year": year,
                "co2_price": point["co2_price"],
                "tech": tech,
                "costs": float(value),
            })
        logger.info(
            "%s: %.2f €/t, atmosphere %.0f Mt, objective %.1f billion euro",
            year,
            point["co2_price"],
            point["co2_mt"],
            point["objective_bn"],
        )

    summary = pd.DataFrame(points)
    combustion = combustion_mt(comparison or (output / "eu_comparison.csv"))
    if not combustion.empty:
        summary["combustion_mt"] = summary["year"].map(combustion)
    costs = pd.DataFrame(cost_rows)
    summary.to_csv(output / "abatement_summary.csv", index=False)
    costs.to_csv(output / "abatement_costs.csv", index=False)
    plot_curve(summary, costs, output / "abatement_curve.png")
    logger.info("Wrote %s", output / "abatement_curve.png")


def run_snakemake(snakemake) -> None:
    """Entry point used by the workflow once the comparison table exists."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    networks = snakemake.input.networks
    if isinstance(networks, str):
        networks = [networks]
    write_curve(
        [Path(path) for path in networks],
        Path(snakemake.output.summary).parent,
        Path(snakemake.input.comparison),
    )


def main() -> None:
    """Read the three solved years and write the EU curve."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "networks",
        nargs="*",
        type=Path,
        default=[
            root / "results/iea-cps/networks/solved_2030.nc",
            root / "results/iea-cps/networks/solved_2035.nc",
            root / "results/iea-cps/networks/solved_2040.nc",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/iea-cps/iea_comparison",
    )
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    write_curve(list(args.networks), args.output, args.comparison)


if __name__ == "__main__":
    if "snakemake" in globals():
        run_snakemake(snakemake)
    else:
        main()
