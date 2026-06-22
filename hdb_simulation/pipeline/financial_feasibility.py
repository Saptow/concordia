"""
financial_feasibility.py
------------------------

Simplified affordability proxy for synthetic HDB resale buyers.

This module estimates a buyer's effective affordability ceiling for use in:
- synthetic market generation
- buyer-flat reachability filtering
- policy scenario testing

Interpretation
--------------
The output is NOT an official HDB / bank eligibility assessment. This is a simplified affordability proxy. 

It is a deterministic affordability proxy based on:

1. Expected working years
2. Estimated accumulated cash savings
3. Estimated CPF-like housing resources
4. Monthly mortgage capacity capped by MSR
5. Loan quantum from standard amortisation
6. Optional grant amount

Main affordability equation
---------------------------

    effective_ceiling =
        estimated_cash_savings
        + cpf_housing_resources
        + loan_quantum
        + grant_amount

This is intended for controlled simulation and scenario comparison, not exact
household-finance modelling.

Key simplifications
-------------------
- Uses one configurable housing loan interest rate.
- Does not model separate HDB vs bank loan packages.
- Does not model COV separately.
- Does not model stamp duties, renovation costs, household liabilities, or exact HFE rules.
- Uses expected working years as a proxy for accumulated savings / CPF resources.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetConfig:
    # -----------------------------------------------------------------------
    # Income / CPF assumptions
    # -----------------------------------------------------------------------

    # CPF Ordinary Wage ceiling.
    # Source: CPF contribution rules.
    # https://www.cpf.gov.sg/employer/employer-obligations/how-much-cpf-contributions-to-pay
    ordinary_wage_ceiling: float = 6_000.0

    # Employee CPF contribution rate, used to estimate take-home income.
    # For this prototype, this is kept as a simple constant.
    employee_cpf_rate: float = 0.20

    # Personal saving rate.
    # Source: SingStat household-sector / personal saving statistics.
    # Fill in the exact source URL used in your README / data documentation.
    personal_saving_rate: float = 0.364

    # -----------------------------------------------------------------------
    # Housing loan assumptions
    # -----------------------------------------------------------------------

    # Single configurable housing-loan interest rate.
    # Default uses HDB concessionary interest rate as a simple benchmark.
    # Source: HDB housing loan from HDB.
    # https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/housing-loan/housing-loan-from-hdb
    housing_loan_interest_rate: float = 0.026

    # Mortgage Servicing Ratio cap.
    # Source: MAS MSR / TDSR rules.
    # https://www.mas.gov.sg/regulation/explainers/new-housing-loans/msr-and-tdsr-rules
    msr_cap: float = 0.30

    # -----------------------------------------------------------------------
    # Loan tenure assumptions
    # -----------------------------------------------------------------------

    # HDB loan tenure rule uses age 65 as one of the caps.
    # This is not meant to model retirement behaviour.
    # Source: HDB housing loan from HDB.
    # https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/housing-loan/housing-loan-from-hdb
    loan_age_cap: int = 65

    # HDB loan tenure cap.
    # Source: HDB housing loan from HDB.
    # https://www.hdb.gov.sg/buying-a-flat/flat-grant-and-loan-eligibility/housing-loan/housing-loan-from-hdb
    max_loan_tenure_years: int = 25


CFG = BudgetConfig()


# ---------------------------------------------------------------------------
# Income band point estimates
# ---------------------------------------------------------------------------

INCOME_BANDS: dict[str, dict[str, float | None]] = {
    "Below 500": {"lower": 0.0, "upper": 500.0},
    "500 - 999": {"lower": 500.0, "upper": 999.0},
    "1,000 - 1,499": {"lower": 1_000.0, "upper": 1_499.0},
    "1,500 - 1,999": {"lower": 1_500.0, "upper": 1_999.0},
    "2,000 - 2,999": {"lower": 2_000.0, "upper": 2_999.0},
    "3,000 - 3,999": {"lower": 3_000.0, "upper": 3_999.0},
    "4,000 - 4,999": {"lower": 4_000.0, "upper": 4_999.0},
    "5,000 - 5,999": {"lower": 5_000.0, "upper": 5_999.0},
    "6,000 - 6,999": {"lower": 6_000.0, "upper": 6_999.0},
    "7,000 - 7,999": {"lower": 7_000.0, "upper": 7_999.0},
    "8,000 - 8,999": {"lower": 8_000.0, "upper": 8_999.0},
    "9,000 - 9,999": {"lower": 9_000.0, "upper": 9_999.0},
    "10,000 & Over": {"lower": 10_000.0, "upper": 15_000.0}, # 15,000 is a reasonable upper bound for this band, since beyond that, people normally would not purchase HDB flats.
}


# ---------------------------------------------------------------------------
# CPF schedules
# ---------------------------------------------------------------------------

CPF_TOTAL_RATE: dict[tuple[int, int], float] = {
    # Source: CPF allocation rates from 1 January 2023.
    # https://www.cpf.gov.sg/content/dam/web/employer/employer-obligations/documents/CPF%20allocation%20rates%20from%201%20January%202023.pdf
    (0, 55): 0.37,
    (56, 60): 0.31,
    (61, 65): 0.22,
    (66, 70): 0.16,
    (71, 200): 0.125,
}


CPF_OA_ALLOCATION: dict[tuple[int, int], float] = {
    # Source: CPF allocation rates from 1 January 2023.
    # https://www.cpf.gov.sg/content/dam/web/employer/employer-obligations/documents/CPF%20allocation%20rates%20from%201%20January%202023.pdf
    (0, 35): 0.6217,
    (36, 45): 0.5677,
    (46, 50): 0.5136,
    (51, 55): 0.4055,
    (56, 60): 0.3871,
    (61, 65): 0.1591,
    (66, 70): 0.0625,
    (71, 200): 0.08,
}


LABOUR_FORCE_PARTICIPATION_TOTAL_2023: dict[tuple[int, int], float] = {
    # Source: Data.gov for resident labour force participation rates in 2023.
    # https://data.gov.sg/datasets/d_465c5bd4a9adae523f9577371daf5e24/view 
    # Uses the 2023 "Total" series only.
    (15, 19): 0.157,
    (20, 24): 0.559,
    (25, 29): 0.885,
    (30, 34): 0.934,
    (35, 39): 0.923,
    (40, 44): 0.905,
    (45, 49): 0.891,
    (50, 54): 0.842,
    (55, 59): 0.771,
    (60, 64): 0.666,
    (65, 69): 0.496,
    (70, 74): 0.329,
    (75, 200): 0.118,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_age_band_value(age: int, table: dict[tuple[int, int], float]) -> float:
    for (lower, upper), value in table.items():
        if lower <= age <= upper:
            return value
    raise ValueError(f"Age {age} is not covered by the rate table.")


def resolve_income_band_upper(
    income_band: str,
    income_bands: dict[str, dict[str, float | None]] = INCOME_BANDS,
) -> float:
    """
    Converts an income band label into a point estimate.

    For this prototype, the upper bound is used to avoid underestimating
    affordability too aggressively during buyer generation. This can be adjusted as needed, especially with HDB's help with more data on affordability. 
    """
    if income_band not in income_bands:
        raise KeyError(
            f"Income band '{income_band}' not found. "
            f"Available bands: {list(income_bands.keys())}"
        )

    upper = income_bands[income_band].get("upper")

    if upper is None:
        raise ValueError(
            f"Income band '{income_band}' has no upper bound. "
            "Please set one explicitly."
        )

    return float(upper)


# ---------------------------------------------------------------------------
# Expected working years
# ---------------------------------------------------------------------------

def estimate_expected_years_worked(current_age: int) -> float:
    """
    Estimate expected years worked from 2023 total labour-force participation.

    Each past age year contributes the 2023 "Total" participation rate for its
    age band. For example, each year from age 20 to 24 contributes 0.559
    expected working years.
    """
    if current_age <= 15:
        return 0.0

    return sum(
        (overlap_end - overlap_start + 1) * participation_rate
        for (band_start, band_end), participation_rate in (
            LABOUR_FORCE_PARTICIPATION_TOTAL_2023.items()
        )
        for overlap_start, overlap_end in [
            (max(15, band_start), min(current_age - 1, band_end))
        ]
        if overlap_start <= overlap_end
    )


# ---------------------------------------------------------------------------
# Step 1: Cash savings proxy
# ---------------------------------------------------------------------------

def estimate_cash_savings(
    current_age: int,
    monthly_income: float,
    cfg: BudgetConfig = CFG,
) -> float:
    """
    Estimates accumulated cash savings.

    Formula:
        estimated_cash_savings =
            monthly_take_home_income
            * 12
            * personal_saving_rate
            * expected_years_worked

    Notes:
    - This is used as part of effective_ceiling.
    - This should not be interpreted as actual liquid wealth. It is a simplified proxy for accumulated savings over working life. 
    """
    expected_years = estimate_expected_years_worked(current_age)
    annual_savings = (
        monthly_income * (1 - cfg.employee_cpf_rate) * 12 * cfg.personal_saving_rate
    )
    return max(0.0, annual_savings * expected_years)


# ---------------------------------------------------------------------------
# Step 2: CPF-like housing resources proxy
# ---------------------------------------------------------------------------

def estimate_cpf_housing_resources(
    current_age: int,
    monthly_income: float,
    cfg: BudgetConfig = CFG,
) -> float:
    """
    Estimates CPF-like housing resources available for purchase.

    Formula:
        cpf_housing_resources =
            annual_oa_contribution_at_current_income
            * expected_years_worked

    This is intentionally simpler than reconstructing exact CPF balances over
    each year of the buyer's working life.
    """
    expected_years = estimate_expected_years_worked(current_age)
    capped_income = min(monthly_income, cfg.ordinary_wage_ceiling)

    annual_oa_contribution = (
        capped_income
        * _get_age_band_value(current_age, CPF_TOTAL_RATE)
        * _get_age_band_value(current_age, CPF_OA_ALLOCATION)
        * 12
    )
    return max(0.0, annual_oa_contribution * expected_years)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuyerFinancials:
    effective_ceiling: float


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def compute_buyer_financials(
    current_age: int,
    monthly_income: float,
    grant_amount: float = 0.0,
    cfg: BudgetConfig = CFG,
) -> BuyerFinancials:
    """
    Computes simplified buyer financial capacity.

    Main affordability equation:
        effective_ceiling =
            estimated_cash_savings
            + cpf_housing_resources
            + loan_quantum
            + grant_amount

    effective_ceiling is left unrounded for computation.
    Round only when displaying or logging outputs.
    """
    if current_age <= 0:
        raise ValueError("current_age must be positive.")

    if monthly_income < 0:
        raise ValueError("monthly_income cannot be negative.")

    if grant_amount < 0:
        raise ValueError("grant_amount cannot be negative.")

    cash_savings = estimate_cash_savings(
        current_age=current_age,
        monthly_income=monthly_income,
        cfg=cfg,
    )

    cpf_resources = estimate_cpf_housing_resources(
        current_age=current_age,
        monthly_income=monthly_income,
        cfg=cfg,
    )

    monthly_mortgage_capacity = max(0.0, monthly_income * cfg.msr_cap)
    tenure_years = max(0, min(cfg.max_loan_tenure_years, cfg.loan_age_cap - current_age))
    if monthly_mortgage_capacity <= 0 or tenure_years <= 0:
        loan_quantum = 0.0
    else:
        monthly_rate = cfg.housing_loan_interest_rate / 12
        n_months = tenure_years * 12
        if monthly_rate == 0:
            loan_quantum = monthly_mortgage_capacity * n_months
        else:
            loan_quantum = (
                monthly_mortgage_capacity
                * (1 - (1 + monthly_rate) ** (-n_months))
                / monthly_rate
            )

    effective_ceiling = cash_savings + cpf_resources + loan_quantum + grant_amount

    return BuyerFinancials(effective_ceiling=round(effective_ceiling, 2))


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_financials_for_pool(
    buyers: list[dict],
    income_bands: dict[str, dict[str, float | None]] = INCOME_BANDS,
    cfg: BudgetConfig = CFG,
    grant_amount: float = 0.0,
) -> list[dict]:
    """
    Adds a 'financials' field to each buyer dict.

    Each buyer dict must contain:
        - age
        - income_band

    Logic:
        monthly_income = upper bound of buyer's income band
        financials.effective_ceiling = simplified affordability proxy
    """
    for buyer in buyers:
        for field in ("age", "income_band"):
            if field not in buyer:
                raise KeyError(f"Buyer record is missing required field: {field!r}.")

        monthly_income = resolve_income_band_upper(
            income_band=buyer["income_band"],
            income_bands=income_bands,
        )

        financials = compute_buyer_financials(
            current_age=int(buyer["age"]),
            monthly_income=monthly_income,
            grant_amount=grant_amount,
            cfg=cfg,
        )

        buyer["financials"] = financials.__dict__

    return buyers
