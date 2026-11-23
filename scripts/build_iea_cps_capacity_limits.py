# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build ``data/agg_p_nom_minmax.csv`` from IEA WEO 2019 Current Policies.

Annex A, Table A.3 (European Union) gives electrical capacity in GW. 2018
is the historical column of that table. 2030 and 2040 are Current Policies.
2035 is the geometric midpoint of that decade. 2045 and 2050 continue the
same 2030-2040 ratio. WEO 2019 stops at 2040, so those two years are an
extrapolation, not a published CPS figure. 2018 is not extrapolated.

The EU total for each technology is shared across the EU-28 countries that
are in this model (Cyprus and Malta are not) in proportion to that
technology's capacity in the powerplant database. Countries outside the 2019
EU (Norway, Switzerland, the Western Balkans) are left unconstrained: the
IEA table is not an EU-plus-neighbours total.

Wind is split into onshore and offshore with the same database. Solar PV and
CSP are one solar total. Hard coal and lignite are one coal total. Gas is
one total for OCGT and CCGT. Hydro is reservoir and run-of-river, not pumped
storage. Bioenergy is solid biomass, biogas and waste. Every technology,
including coal and oil, is fixed at the country share of the EU total.
Nuclear uses the capacity still licensed in that year, so its country mix
changes as units close.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.nuclear_fleet import (
    apply_published_nuclear_dates,
    fill_missing_nuclear_dateout,
    nuclear_online_capacity,
)

logger = logging.getLogger(__name__)

# IEA WEO 2019, Annex A, Table A.3, European Union, GW.
# 2018 is the historical column. 2035, 2045 and 2050 are filled in
# ``_extend`` from the 2030-2040 ratio and do not use 2018.

HORIZONS = [2018, 2030, 2035, 2040, 2045, 2050]

# EU-28 as defined in WEO 2019, minus Cyprus and Malta (not in the model).
EU_IN_MODEL = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
    "GB",
}

NAME_TO_ISO = {
    "Austria": "AT",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Croatia": "HR",
    "Czechia": "CZ",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "GR",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Netherlands": "NL",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Spain": "ES",
    "Sweden": "SE",
    "United Kingdom": "GB",
}


def _extend(
    gw_2030: float, gw_2040: float, gw_2018: float | None = None
) -> dict[int, float]:
    """Fill 2035, 2045 and 2050 from the published 2030 and 2040 values.

    2018 is the historical column when it is passed in. It does not change
    the 2030-2040 ratio.
    """
    ratio = gw_2040 / gw_2030
    step = ratio**0.5
    years = {
        2030: gw_2030,
        2035: gw_2030 * step,
        2040: gw_2040,
        2045: gw_2040 * step,
        2050: gw_2040 * ratio,
    }
    if gw_2018 is not None:
        years = {2018: gw_2018, **years}
    return years


def _shares(plants: pd.DataFrame, mask: pd.Series) -> pd.Series:
    cap = plants.loc[mask].groupby("iso")["Capacity"].sum()
    cap = cap.reindex(sorted(EU_IN_MODEL)).fillna(0.0)
    total = cap.sum()
    if total <= 0:
        raise ValueError("No capacity found for a technology share.")
    return cap / total


def build_limits(plants: pd.DataFrame) -> pd.DataFrame:
    plants = plants.copy()
    plants["iso"] = plants["Country"].map(NAME_TO_ISO)
    plants = plants.dropna(subset=["iso"])
    plants = plants[plants["iso"].isin(EU_IN_MODEL)]

    wind = plants["Fueltype"].eq("Wind")
    onshore = wind & plants["Technology"].eq("Onshore")
    offshore = wind & plants["Technology"].eq("Offshore")
    solar = plants["Fueltype"].eq("Solar") & plants["Technology"].str.upper().eq("PV")
    # CSP is small and sits in the same solar cap as PV.
    solar = solar | (
        plants["Fueltype"].eq("Solar") & plants["Technology"].str.upper().eq("CSP")
    )
    gas = plants["Fueltype"].eq("Natural Gas")
    coal = plants["Fueltype"].isin(["Hard Coal", "Lignite"])
    oil = plants["Fueltype"].eq("Oil")
    hydro = plants["Fueltype"].eq("Hydro") & plants["Technology"].isin(
        ["Reservoir", "Run-Of-River"]
    )
    biomass = plants["Fueltype"].isin(["Solid Biomass", "Biogas", "Waste"])
    geothermal = plants["Fueltype"].eq("Geothermal")

    wind_on_share = _shares(plants, onshore)
    wind_off_share = _shares(plants, offshore)
    on_gw = plants.loc[onshore, "Capacity"].sum()
    off_gw = plants.loc[offshore, "Capacity"].sum()
    on_frac = on_gw / (on_gw + off_gw)

    # 2018 solar is 115 GW of PV plus 2 GW of CSP, one solar total.
    paths = {
        "onwind": _extend(286.0 * on_frac, 321.0 * on_frac, 180.0 * on_frac),
        "offwind-all": _extend(
            286.0 * (1 - on_frac), 321.0 * (1 - on_frac), 180.0 * (1 - on_frac)
        ),
        "solar-all": _extend(213.0, 240.0, 117.0),
        "gas": _extend(304.0, 345.0, 217.0),
        "coal-all": _extend(112.0, 66.0, 164.0),
        "oil": _extend(20.0, 13.0, 45.0),
    }
    optional = {
        "hydro": (hydro, _extend(163.0, 168.0, 155.0)),
        "biomass": (biomass, _extend(56.0, 59.0, 48.0)),
        "geothermal": (geothermal, _extend(1.0, 2.0, 1.0)),
    }
    country_share = {
        "onwind": wind_on_share,
        "offwind-all": wind_off_share,
        "solar-all": _shares(plants, solar),
        "gas": _shares(plants, gas),
        "coal-all": _shares(plants, coal),
        "oil": _shares(plants, oil),
    }
    for carrier, (mask, years) in optional.items():
        if mask.any() and float(plants.loc[mask, "Capacity"].sum()) > 0:
            paths[carrier] = years
            country_share[carrier] = _shares(plants, mask)
    # Fixed at the CPS total, so the optimum cannot sit below the IEA capacity.
    floor = set(paths) | {"nuclear"}
    nuclear_rows = _nuclear_rows(plants)

    rows = []
    for carrier, years in paths.items():
        for country, share in country_share[carrier].items():
            row = {"country": country, "carrier": carrier}
            for year, gw in years.items():
                mw = round(gw * 1000 * share, 1)
                row[(year, "min")] = mw if carrier in floor else pd.NA
                row[(year, "max")] = mw
            rows.append(row)
    rows.extend(nuclear_rows)
    frame = pd.DataFrame(rows).set_index(["country", "carrier"])
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame.sort_index()


def _nuclear_rows(plants: pd.DataFrame) -> list[dict]:
    """Fix each EU country at its share of the CPS nuclear total."""
    required = {"Fueltype", "Country", "Capacity", "Name", "DateIn", "DateOut"}
    if not required <= set(plants.columns):
        return []
    nuclear = plants["Fueltype"].astype(str).str.lower().eq("nuclear")
    if not nuclear.any():
        return []
    dated = plants.loc[nuclear].copy()
    dated["Country"] = dated["Country"].map(NAME_TO_ISO)
    dated = dated.dropna(subset=["Country"])
    dated = dated[dated["Country"].isin(EU_IN_MODEL)]
    if dated.empty:
        return []
    dated = fill_missing_nuclear_dateout(dated)
    dated = apply_published_nuclear_dates(dated)
    dated = fill_missing_nuclear_dateout(dated)
    # WEO 2019 Table A.3, European Union. Historical capacity in 2018 is
    # 125 GW. Current Policies is 97 GW in 2030 and 92 GW in 2040.
    targets = _extend(97.0, 92.0, 125.0)
    rows = []
    for year, gw in targets.items():
        online = nuclear_online_capacity(dated, year)
        online = online.reindex(sorted(EU_IN_MODEL)).fillna(0.0)
        total = float(online.sum())
        shares = online / total if total > 0 else online * 0.0
        for country, share in shares.items():
            mw = round(float(gw) * 1000 * float(share), 1)
            row = {"country": country, "carrier": "nuclear"}
            # Find and update an existing row for this country, or start one.
            existing = next(
                (item for item in rows if item["country"] == country), None
            )
            if existing is None:
                existing = row
                rows.append(existing)
            existing[(year, "min")] = mw
            existing[(year, "max")] = mw
    return rows


def carrier_limit(path: str | Path, carrier: str, year: int) -> pd.Series:
    """Electrical MW cap for one carrier and planning year, indexed by country."""
    table = pd.read_csv(path, index_col=[0, 1], header=[0, 1])
    if carrier not in set(table.index.get_level_values(1)):
        return pd.Series(dtype=float)
    years = set(table.columns.get_level_values(0))
    key: str | int | None = None
    for candidate in (str(year), year):
        if candidate in years:
            key = candidate
            break
    if key is None:
        raise KeyError(
            f"No {carrier} limit for {year}. Years in {path}: "
            f"{sorted({str(item) for item in years})}."
        )
    values = table[key].xs(carrier, level=1)["max"].astype(float)
    values.index = values.index.astype(str)
    return values


def write_limits(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Header layout read by solve_network.add_CCL_constraints.
    years = HORIZONS
    header = [""] * 2 + [year for year in years for _ in ("min", "max")]
    header2 = ["country", "carrier"] + ["min", "max"] * len(years)
    lines = [
        ",".join(str(item) for item in header),
        ",".join(header2),
    ]
    for (country, carrier), row in frame.iterrows():
        values = []
        for year in years:
            for bound in ("min", "max"):
                value = row[(year, bound)]
                values.append("" if pd.isna(value) else f"{value:.1f}")
        lines.append(",".join([country, carrier, *values]))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("powerplants", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/agg_p_nom_minmax.csv"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plants = pd.read_csv(
        args.powerplants,
        usecols=[
            "Fueltype",
            "Technology",
            "Country",
            "Capacity",
            "Name",
            "DateIn",
            "DateOut",
            "DateRetrofit",
        ],
    )
    frame = build_limits(plants)
    write_limits(frame, args.output)
    totals = frame.xs("max", level=1, axis=1).groupby(level=1).sum() / 1000
    logger.info("\n%s", totals.round(2).to_string())
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
