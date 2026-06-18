import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import os

import pandas as pd


# -----------------------------------------------------------------------------
# Core model inputs
# -----------------------------------------------------------------------------

@dataclass
class GasExchangeInputs:
    """
    Geometry + transport properties for an open vertical vessel initially
    flushed with nitrogen and then exposed to ambient air.

    H_m: vessel height [m]
    R_m: vessel radius [m]
    """
    H_m: float
    R_m: float

    # Transport / fluid properties at room temperature
    D_m2_s: float = 2.0e-5      # O2 in N2 diffusion coefficient [m^2/s]
    nu_m2_s: float = 1.5e-5     # kinematic viscosity [m^2/s]
    g_m_s2: float = 9.81
    ambient_o2_percent: float = 20.95

    # Density contrast: air vs N2, dimensionless
    delta_rho_over_rho: float = 0.034

    # Heuristic convection model
    # We model real-world contamination as first-order approach to ambient:
    #   F(t) = 1 - exp(-t / tau)
    # and calibrate tau from t95 (time to 95% of ambient contamination).
    #
    # t95 is estimated from a turnover argument:
    #   t95 ~ n_turnovers * H / U
    # with U ~ c * sqrt(g' * Lc), g' = g * delta_rho/rho.
    #
    # "fast" = stronger buoyancy/disturbance, more conservative for process design
    # "slow" = quieter case
    c_conv_fast: float = 0.05
    c_conv_slow: float = 0.01
    n_turnovers_fast: float = 3.0
    n_turnovers_slow: float = 10.0

    # Numerical series truncation for the exact 1D diffusion solution
    n_terms: int = 200


# -----------------------------------------------------------------------------
# Unit/spec conversions
# -----------------------------------------------------------------------------

def abs_o2_percent_to_fraction_of_ambient(
    abs_o2_percent: float, ambient_o2_percent: float = 20.95
) -> float:
    """
    Convert absolute oxygen concentration in the vessel, e.g. 1.0% O2,
    into fraction of ambient oxygen concentration.
    """
    if abs_o2_percent < 0:
        raise ValueError("abs_o2_percent must be non-negative.")
    return abs_o2_percent / ambient_o2_percent


def fraction_of_ambient_to_abs_o2_percent(
    fraction_of_ambient: float, ambient_o2_percent: float = 20.95
) -> float:
    """
    Convert fraction of ambient oxygen concentration into absolute %O2.
    """
    if fraction_of_ambient < 0:
        raise ValueError("fraction_of_ambient must be non-negative.")
    return fraction_of_ambient * ambient_o2_percent


# -----------------------------------------------------------------------------
# Exact 1D diffusion-only solution
# -----------------------------------------------------------------------------

def avg_fraction_diffusion(
    t_s: float,
    H_m: float,
    D_m2_s: float,
    n_terms: int = 200,
) -> float:
    """
    Exact average contamination fraction relative to ambient oxygen for the 1D model:

        dC/dt = D d2C/dz2, 0<z<H
        C(0,t) = C0
        dC/dz(H,t) = 0
        C(z,0) = 0

    Returns avg(C)/C0 in [0, 1].
    """
    s = 0.0
    for n in range(n_terms):
        m = 2 * n + 1
        s += (
            8.0 / (m * m * math.pi * math.pi)
            * math.exp(-(m * m * math.pi * math.pi * D_m2_s * t_s) / (4.0 * H_m * H_m))
        )
    return max(0.0, min(1.0, 1.0 - s))


def time_to_fraction_diffusion(
    target_fraction_of_ambient: float,
    H_m: float,
    D_m2_s: float,
    n_terms: int = 200,
    t_upper_guess_s: Optional[float] = None,
) -> float:
    """
    Solve for time when average oxygen fraction reaches target_fraction_of_ambient.
    """
    if not (0.0 < target_fraction_of_ambient < 1.0):
        raise ValueError("target_fraction_of_ambient must be between 0 and 1.")

    if t_upper_guess_s is None:
        t_upper_guess_s = 20.0 * H_m * H_m / D_m2_s

    lo, hi = 0.0, t_upper_guess_s

    while avg_fraction_diffusion(hi, H_m, D_m2_s, n_terms) < target_fraction_of_ambient:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("Failed to bracket diffusion time.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        val = avg_fraction_diffusion(mid, H_m, D_m2_s, n_terms)
        if val < target_fraction_of_ambient:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# -----------------------------------------------------------------------------
# Convection screening + engineering first-order contamination model
# -----------------------------------------------------------------------------

def solutal_rayleigh(
    length_scale_m: float,
    delta_rho_over_rho: float,
    g_m_s2: float,
    nu_m2_s: float,
    D_m2_s: float,
) -> float:
    return g_m_s2 * delta_rho_over_rho * length_scale_m**3 / (nu_m2_s * D_m2_s)


def engineering_envelope(inputs: GasExchangeInputs) -> Dict[str, float]:
    """
    Compute buoyancy screening and heuristic t95 contamination times for:
      - fast case (conservative)
      - slow case (quieter process)
    """
    H = inputs.H_m
    R = inputs.R_m
    d = 2.0 * R

    g_reduced = inputs.g_m_s2 * inputs.delta_rho_over_rho

    Ra_H = solutal_rayleigh(H, inputs.delta_rho_over_rho, inputs.g_m_s2, inputs.nu_m2_s, inputs.D_m2_s)
    Ra_d = solutal_rayleigh(d, inputs.delta_rho_over_rho, inputs.g_m_s2, inputs.nu_m2_s, inputs.D_m2_s)
    Ra_R = solutal_rayleigh(R, inputs.delta_rho_over_rho, inputs.g_m_s2, inputs.nu_m2_s, inputs.D_m2_s)

    # Conservative circulation length for small vessels
    Lc = min(H, d)

    U_fast = inputs.c_conv_fast * math.sqrt(g_reduced * Lc)
    U_slow = inputs.c_conv_slow * math.sqrt(g_reduced * Lc)

    # Heuristic time to 95% of ambient contamination
    t95_fast = inputs.n_turnovers_fast * H / max(U_fast, 1e-12)
    t95_slow = inputs.n_turnovers_slow * H / max(U_slow, 1e-12)

    return {
        "g_reduced_m_s2": g_reduced,
        "Ra_H": Ra_H,
        "Ra_diameter": Ra_d,
        "Ra_radius": Ra_R,
        "U_fast_m_s": U_fast,
        "U_slow_m_s": U_slow,
        "t95_fast_s": t95_fast,   # conservative contamination envelope
        "t95_slow_s": t95_slow,
    }


def time_to_fraction_first_order_from_t95(
    target_fraction_of_ambient: float,
    t95_s: float,
) -> float:
    """
    First-order mixing model:
        F(t) = 1 - exp(-t / tau)
    with calibration F(t95) = 0.95.

    Then:
        tau = t95 / ln(20)
        t(F) = -tau * ln(1 - F)
    """
    if not (0.0 < target_fraction_of_ambient < 1.0):
        raise ValueError("target_fraction_of_ambient must be between 0 and 1.")
    tau = t95_s / math.log(20.0)
    return -tau * math.log(1.0 - target_fraction_of_ambient)


# -----------------------------------------------------------------------------
# Process recommendation logic
# -----------------------------------------------------------------------------

def recommend_max_delay(
    inputs: GasExchangeInputs,
    max_allowed_o2_percent_abs: float,
    safety_factor: float = 0.5,
) -> Dict[str, float]:
    """
    Recommend maximum delay between N2 flush and seal for a chosen oxygen spec.

    max_allowed_o2_percent_abs:
        maximum allowed average oxygen concentration inside vessel, in absolute %O2
        e.g. 1.0 means 1% O2 absolute inside the headspace.

    safety_factor:
        multiply the conservative time by this factor.
        Example:
            0.5 -> use half the predicted conservative contamination time

    Returns diffusion, fast-envelope, slow-envelope times to spec and recommended max delay.
    """
    if not (0.0 < max_allowed_o2_percent_abs < inputs.ambient_o2_percent):
        raise ValueError("max_allowed_o2_percent_abs must be between 0 and ambient oxygen percent.")

    if not (0.0 < safety_factor <= 1.0):
        raise ValueError("safety_factor must be in (0, 1].")

    target_fraction = abs_o2_percent_to_fraction_of_ambient(
        max_allowed_o2_percent_abs, inputs.ambient_o2_percent
    )

    env = engineering_envelope(inputs)

    t_spec_diff = time_to_fraction_diffusion(
        target_fraction_of_ambient=target_fraction,
        H_m=inputs.H_m,
        D_m2_s=inputs.D_m2_s,
        n_terms=inputs.n_terms,
    )

    t_spec_fast = time_to_fraction_first_order_from_t95(
        target_fraction_of_ambient=target_fraction,
        t95_s=env["t95_fast_s"],
    )

    t_spec_slow = time_to_fraction_first_order_from_t95(
        target_fraction_of_ambient=target_fraction,
        t95_s=env["t95_slow_s"],
    )

    # Conservative process recommendation:
    # choose the fastest contamination mechanism and apply safety factor
    t_recommended = safety_factor * min(t_spec_diff, t_spec_fast)

    return {
        "target_o2_percent_abs": max_allowed_o2_percent_abs,
        "target_fraction_of_ambient": target_fraction,
        "time_to_spec_diffusion_s": t_spec_diff,
        "time_to_spec_fast_envelope_s": t_spec_fast,
        "time_to_spec_slow_envelope_s": t_spec_slow,
        "recommended_max_delay_s": t_recommended,
    }


# -----------------------------------------------------------------------------
# DataFrame / CSV export
# -----------------------------------------------------------------------------

def summarize_case(
    inputs: GasExchangeInputs,
    oxygen_specs_abs_percent: List[float],
    safety_factor: float = 0.5,
) -> Dict[str, float]:
    """
    Flatten one geometry case into a single dict suitable for a pandas row.
    """
    env = engineering_envelope(inputs)

    row = {
        "H_mm": 1000.0 * inputs.H_m,
        "R_mm": 1000.0 * inputs.R_m,
        "diameter_mm": 2000.0 * inputs.R_m,
        "volume_mL": math.pi * inputs.R_m**2 * inputs.H_m * 1e6,  # m^3 -> mL
        "g_reduced_m_s2": env["g_reduced_m_s2"],
        "Ra_H": env["Ra_H"],
        "Ra_diameter": env["Ra_diameter"],
        "Ra_radius": env["Ra_radius"],
        "Ra_min": min(env["Ra_H"], env["Ra_diameter"], env["Ra_radius"]),
        "U_fast_m_s": env["U_fast_m_s"],
        "U_slow_m_s": env["U_slow_m_s"],
        "t95_fast_s": env["t95_fast_s"],
        "t95_slow_s": env["t95_slow_s"],
    }

    for spec_abs in oxygen_specs_abs_percent:
        rec = recommend_max_delay(
            inputs=inputs,
            max_allowed_o2_percent_abs=spec_abs,
            safety_factor=safety_factor,
        )
        tag = f"{spec_abs:.2f}".replace(".", "p")
        row[f"t_spec_diff_{tag}_pctO2_s"] = rec["time_to_spec_diffusion_s"]
        row[f"t_spec_fast_{tag}_pctO2_s"] = rec["time_to_spec_fast_envelope_s"]
        row[f"t_spec_slow_{tag}_pctO2_s"] = rec["time_to_spec_slow_envelope_s"]
        row[f"recommended_max_delay_{tag}_pctO2_s"] = rec["recommended_max_delay_s"]

    return row


def sweep_to_dataframe(
    H_values_mm: List[float],
    R_values_mm: List[float],
    oxygen_specs_abs_percent: List[float],
    safety_factor: float = 0.5,
    template_inputs: Optional[GasExchangeInputs] = None,
) -> pd.DataFrame:
    """
    Sweep geometry range and return a pandas DataFrame.
    """
    if template_inputs is None:
        template_inputs = GasExchangeInputs(H_m=0.05, R_m=0.01)

    rows = []
    for H_mm in H_values_mm:
        for R_mm in R_values_mm:
            inputs = GasExchangeInputs(
                H_m=H_mm / 1000.0,
                R_m=R_mm / 1000.0,
                D_m2_s=template_inputs.D_m2_s,
                nu_m2_s=template_inputs.nu_m2_s,
                g_m_s2=template_inputs.g_m_s2,
                ambient_o2_percent=template_inputs.ambient_o2_percent,
                delta_rho_over_rho=template_inputs.delta_rho_over_rho,
                c_conv_fast=template_inputs.c_conv_fast,
                c_conv_slow=template_inputs.c_conv_slow,
                n_turnovers_fast=template_inputs.n_turnovers_fast,
                n_turnovers_slow=template_inputs.n_turnovers_slow,
                n_terms=template_inputs.n_terms,
            )
            rows.append(
                summarize_case(
                    inputs=inputs,
                    oxygen_specs_abs_percent=oxygen_specs_abs_percent,
                    safety_factor=safety_factor,
                )
            )

    df = pd.DataFrame(rows)

    # Useful sort: worst-case shortest recommended delay first
    rec_cols = [c for c in df.columns if c.startswith("recommended_max_delay_")]
    if rec_cols:
        df = df.sort_values(by=rec_cols[0], ascending=True).reset_index(drop=True)

    return df


def export_results_csv(
    df: pd.DataFrame,
    csv_path: str,
    excel_friendly: bool = True,
) -> None:
    """
    Export DataFrame to CSV.
    """
    sep = ";" if excel_friendly else ","
    df.to_csv(csv_path, index=False, sep=sep)


# -----------------------------------------------------------------------------
# Pretty printing helpers
# -----------------------------------------------------------------------------

def print_case_report(
    inputs: GasExchangeInputs,
    oxygen_specs_abs_percent: List[float],
    safety_factor: float = 0.5,
) -> None:
    env = engineering_envelope(inputs)
    volume_mL = math.pi * inputs.R_m**2 * inputs.H_m * 1e6

    print("=" * 90)
    print(
        f"H = {1000*inputs.H_m:.1f} mm, "
        f"R = {1000*inputs.R_m:.1f} mm, "
        f"D = {2000*inputs.R_m:.1f} mm, "
        f"V = {volume_mL:.2f} mL"
    )
    print(f"Reduced gravity g'      = {env['g_reduced_m_s2']:.4f} m/s^2")
    print(f"Ra(H)                   = {env['Ra_H']:.3e}")
    print(f"Ra(diameter)            = {env['Ra_diameter']:.3e}")
    print(f"Ra(radius)              = {env['Ra_radius']:.3e}")
    print(f"U_fast                  = {env['U_fast_m_s']:.4f} m/s")
    print(f"U_slow                  = {env['U_slow_m_s']:.4f} m/s")
    print(f"t95 fast envelope       = {env['t95_fast_s']:.1f} s")
    print(f"t95 slow envelope       = {env['t95_slow_s']:.1f} s")

    print("\nSpec-based recommended max delay (flush -> seal):")
    for spec in oxygen_specs_abs_percent:
        rec = recommend_max_delay(inputs, spec, safety_factor=safety_factor)
        print(
            f"  Spec <= {spec:.2f}% O2: "
            f"diff={rec['time_to_spec_diffusion_s']:.1f}s, "
            f"fast={rec['time_to_spec_fast_envelope_s']:.1f}s, "
            f"slow={rec['time_to_spec_slow_envelope_s']:.1f}s, "
            f"RECOMMENDED={rec['recommended_max_delay_s']:.1f}s"
        )
    print("=" * 90)
    print()


# -----------------------------------------------------------------------------
# Example usage for geometry range
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Geometry range from your colleague:
    # height = 45 to 61 mm
    # radius = 9 to 18 mm
    H_values_mm = [45, 50, 55, 61]
    R_values_mm = [9, 12, 15, 18]

    # Choose process specs in absolute %O2 inside vessel
    oxygen_specs_abs_percent = [0.5, 1.0, 2.0]

    # Safety factor for recommended max delay
    # 0.5 = use half of the conservative predicted contamination time
    safety_factor = 0.5

    # Optional: tune heuristics if you know your line is quieter or more disturbed
    template = GasExchangeInputs(
        H_m=0.05,
        R_m=0.01,
        D_m2_s=2.0e-5,
        nu_m2_s=1.5e-5,
        delta_rho_over_rho=0.034,
        c_conv_fast=0.05,
        c_conv_slow=0.01,
        n_turnovers_fast=3.0,
        n_turnovers_slow=10.0,
    )

    # Print one example case
    example_case = GasExchangeInputs(H_m=0.061, R_m=0.018, **{
        k: v for k, v in asdict(template).items() if k not in {"H_m", "R_m"}
    })
    print_case_report(example_case, oxygen_specs_abs_percent, safety_factor=safety_factor)

    # Sweep entire range into DataFrame
    df = sweep_to_dataframe(
        H_values_mm=H_values_mm,
        R_values_mm=R_values_mm,
        oxygen_specs_abs_percent=oxygen_specs_abs_percent,
        safety_factor=safety_factor,
        template_inputs=template,
    )

    # Show compact summary in console
    cols_to_show = [
        "H_mm",
        "R_mm",
        "diameter_mm",
        "volume_mL",
        "Ra_min",
        "t95_fast_s",
        "t95_slow_s",
        "recommended_max_delay_0p50_pctO2_s",
        "recommended_max_delay_1p00_pctO2_s",
        "recommended_max_delay_2p00_pctO2_s",
    ]
    print(df[cols_to_show].to_string(index=False))

    # Export CSV for Excel / process review
    # Create the path by joining the folder name and the file name
    save_path = os.path.join("Oxygen calculation", "nitrogen_flush_delay_estimates.csv")
    export_results_csv(df, save_path, excel_friendly=True)
    print("\nSaved: nitrogen_flush_delay_estimates.csv")
	
	
###############################################################

# A few operational notes:

# recommended_max_delay_* is the one you would use for line design.
# It is computed from the faster contamination mechanism and then multiplied by a safety factor.
# For production use, the most important inputs to tune are:
# c_conv_fast, c_conv_slow
# n_turnovers_fast, n_turnovers_slow 
# safety_factor

# A sensible starting point is:
# safety_factor = 0.5
# use the 1.0% O2 or 0.5% O2 spec column as your design criterion.

