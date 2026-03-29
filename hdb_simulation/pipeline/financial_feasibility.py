"""
financial_feasibility.py
------------------------
Estimates financial capacity parameters for synthetic HDB resale buyers.

Interpretation
--------------
Outputs are deterministic affordability proxies based on:
1. CPF OA accumulation from CPF contribution/allocation schedules
2. Cash savings from a single published personal saving rate
3. Monthly debt-service capacity capped by:
   - MSR
   - TDSR
   - MAS-published 43% median TDSR benchmark for new mortgages
4. Loan quantum from standard amortisation
5. Resale cash allocation between downpayment and COV
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetConfig:
    ow_ceiling: float = 6_000.0                  # CPF Ordinary Wage ceiling
    employee_cpf_rate: float = 0.20             # employee CPF contribution
    cpf_oa_interest: float = 0.025              # OA interest, annual
    personal_saving_rate: float = 0.364         # SingStat personal saving rate
    debt_service_benchmark_ratio: float = 0.43  # MAS median TDSR for new mortgages (see https://www.mas.gov.sg/news/parliamentary-replies/2022/reply-to-parliamentary-question-on-risk-assessment-of-borrowers-defaulting-on-loans-financed-through-floating-rate-loan-packages-by-local-banks )

    hdb_income_ceiling: float = 14_000.0
    hdb_rate: float = 0.026
    bank_rate: float = 0.038

    msr_cap: float = 0.30
    tdsr_cap: float = 0.55

    hdb_ltv: float = 0.80                      
    bank_ltv: float = 0.75
    bank_min_cash_dp: float = 0.05

    working_start_age: int = 22
    retirement_age: int = 65
    max_loan_tenure_years: int = 25


CFG = BudgetConfig()
# ---------------------------------------------------------------------------
# Income band point estimates
# ---------------------------------------------------------------------------

INCOME_BANDS: dict[str, dict[str, float | None]] = {
    "Below $3,000": {"lower": 0.0, "upper": 3000.0},
    "$3,000-$4,999": {"lower": 3000.0, "upper": 4999.0},
    "$5,000-$6,999": {"lower": 5000.0, "upper": 6999.0},
    "$7,000-$9,999": {"lower": 7000.0, "upper": 9999.0},
    "$10,000-$14,999": {"lower": 10000.0, "upper": 14999.0},
    "$15,000-$19,999": {"lower": 15000.0, "upper": 19999.0},
    "$20,000 and above": {"lower": 20000.0, "upper": 250000.0},  # FIXED Assumption 
}

# ---------------------------------------------------------------------------
# CPF schedules
# ---------------------------------------------------------------------------

CPF_TOTAL_RATE: dict[tuple[int, int], float] = { # Source: https://www.cpf.gov.sg/content/dam/web/employer/employer-obligations/documents/CPF%20allocation%20rates%20from%201%20January%202023.pdf
    (0, 55): 0.37,
    (56, 60): 0.31,
    (61, 65): 0.22,
    (66, 70): 0.16,
    (71, 200): 0.125,
}

CPF_OA_ALLOCATION: dict[tuple[int, int], float] = {
    (0, 35): 0.6217,
    (36, 45): 0.5677,
    (46, 50): 0.5136,
    (51, 55): 0.4055,
    (56, 60): 0.3871,
    (61, 65): 0.1591,
    (66, 70): 0.0625,
    (71, 200): 0.08,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_band_value(age: int, table: dict[tuple[int, int], float]) -> float:
    for (lo, hi), value in table.items():
        if lo <= age <= hi:
            return value
    raise ValueError(f"Age {age} not covered by rate table.")


def resolve_income_band_upper(
    income_band: str,
    income_bands: dict[str, dict[str, float | None]] = INCOME_BANDS,
) -> float:
    if income_band not in income_bands:
        raise KeyError(
            f"Income band '{income_band}' not found. "
            f"Available bands: {list(income_bands.keys())}"
        )

    upper = income_bands[income_band].get("upper")
    if upper is None:
        raise ValueError(
            f"Income band '{income_band}' has no upper bound. "
            f"Please set one explicitly."
        )

    return float(upper)


# ---------------------------------------------------------------------------
# Step 1: CPF OA balance
# ---------------------------------------------------------------------------

def estimate_cpf_oa_balance(
    current_age: int,
    monthly_income: float,
    cfg: BudgetConfig = CFG,
) -> float:
    """
    Deterministic CPF OA accumulation from working_start_age to current_age - 1.

    Assumptions:
    - income is constant in real terms over the working period
    - monthly CPF income is capped at the OW ceiling
    - OA balance compounds annually at the OA interest rate
    """
    if current_age <= cfg.working_start_age:
        return 0.0

    capped_income = min(monthly_income, cfg.ow_ceiling)
    balance = 0.0

    for age in range(cfg.working_start_age, current_age):
        total_rate = _get_band_value(age, CPF_TOTAL_RATE)
        oa_alloc = _get_band_value(age, CPF_OA_ALLOCATION)
        annual_oa = capped_income * total_rate * oa_alloc * 12
        balance = balance * (1 + cfg.cpf_oa_interest) + annual_oa

    return max(0.0, balance)


# ---------------------------------------------------------------------------
# Step 2: Cash savings
# ---------------------------------------------------------------------------

def estimate_cash_savings(
    current_age: int,
    monthly_income: float,
    cfg: BudgetConfig = CFG,
) -> float:
    years_working = max(0, current_age - cfg.working_start_age)
    monthly_disposable = monthly_income * (1 - cfg.employee_cpf_rate)
    annual_savings = monthly_disposable * 12 * cfg.personal_saving_rate

    balance = 0.0
    for _ in range(years_working):
        balance += annual_savings # assume no interest on cash savings for simplicity
    return max(0.0, balance)


# ---------------------------------------------------------------------------
# Step 3: Monthly debt-service capacity
# ---------------------------------------------------------------------------

def compute_max_monthly_mortgage(
    monthly_income: float,
    cfg: BudgetConfig = CFG,
) -> tuple[float, float]:
    """
    Returns:
        (max_monthly_mortgage, total_debt_service_benchmark_monthly)

    The 43% figure is treated as an all-in debt-service benchmark, not
    pre-existing debt. This avoids double counting housing debt.
    """
    msr_limit = monthly_income * cfg.msr_cap
    tdsr_limit = monthly_income * cfg.tdsr_cap
    debt_service_benchmark = monthly_income * cfg.debt_service_benchmark_ratio

    max_monthly_mortgage = max(
        0.0,
        min(msr_limit, tdsr_limit, debt_service_benchmark),
    )
    return max_monthly_mortgage, debt_service_benchmark


# ---------------------------------------------------------------------------
# Step 4: Max loan quantum
# ---------------------------------------------------------------------------

def compute_max_loan_quantum(
    max_monthly_payment: float,
    use_hdb_loan: bool,
    current_age: int,
    cfg: BudgetConfig = CFG,
) -> tuple[float, int]:
    """Returns (max_loan_quantum, effective_tenure_years)."""
    tenure_years = min(cfg.max_loan_tenure_years, cfg.retirement_age - current_age)
    tenure_years = max(0, tenure_years)

    if tenure_years == 0 or max_monthly_payment <= 0:
        return 0.0, 0

    annual_rate = cfg.hdb_rate if use_hdb_loan else cfg.bank_rate
    monthly_rate = annual_rate / 12
    n_months = tenure_years * 12

    pv = max_monthly_payment * (1 - (1 + monthly_rate) ** (-n_months)) / monthly_rate
    return max(0.0, pv), tenure_years


# ---------------------------------------------------------------------------
# Step 5: Resale cash allocation
# ---------------------------------------------------------------------------

def allocate_resale_cash(
    loan_quantum: float,
    cpf_oa_balance: float,
    cash_savings: float,
    use_hdb_loan: bool,
    forced_max_cov: float = 0.0,
    cfg: BudgetConfig = CFG,
) -> tuple[float, float]:
    """
    Returns:
        (max_valuation, max_cov)

    HDB loan:
        downpayment = 20% of valuation
        CPF OA can cover the full 20%
        cash is needed only for any CPF shortfall
        residual cash can go to COV

    Bank loan:
        downpayment = 25% of valuation
        5% must be cash
        CPF OA can cover up to the remaining 20%
        residual cash can go to COV
    """
    ltv = cfg.hdb_ltv if use_hdb_loan else cfg.bank_ltv
    val_from_loan = (loan_quantum / ltv) if ltv > 0 else 0.0

    if use_hdb_loan:
        cash_needed_for_dp = max(0.0, (1 - ltv) * val_from_loan - cpf_oa_balance)
    else:
        min_cash_dp = cfg.bank_min_cash_dp * val_from_loan
        cpf_cover_limit = (1 - ltv - cfg.bank_min_cash_dp) * val_from_loan
        cpf_usable_dp = min(cpf_oa_balance, cpf_cover_limit)
        remaining_dp = max(0.0, (1 - ltv) * val_from_loan - min_cash_dp - cpf_usable_dp)
        cash_needed_for_dp = min_cash_dp + remaining_dp

    if cash_savings >= cash_needed_for_dp:
        max_valuation = val_from_loan
        max_cov = cash_savings - cash_needed_for_dp
    else:
        if use_hdb_loan:
            max_valuation = (cash_savings + cpf_oa_balance) / (1 - ltv)
        else:
            cpf_cover_limit = (1 - ltv - cfg.bank_min_cash_dp) * val_from_loan
            if cpf_oa_balance >= cpf_cover_limit:
                max_valuation = cash_savings / cfg.bank_min_cash_dp
            else:
                max_valuation = (cash_savings + cpf_oa_balance) / (1 - ltv)
        max_cov = 0.0

    max_valuation = min(max_valuation, val_from_loan)
    max_cov = forced_max_cov

    return max(0.0, max_valuation), max(0.0, max_cov)



# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuyerFinancials:
    max_valuation: float
    max_cov: float
    effective_ceiling: float

    cpf_oa_balance: float
    cash_savings: float
    loan_quantum: float
    loan_type: Literal["hdb", "bank"]
    monthly_mortgage_capacity: float
    effective_tenure_years: int
    monthly_income: float
    total_debt_service_benchmark_monthly: float


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def compute_buyer_financials(
    current_age: int,
    monthly_income: float,
    forced_max_cov: float,
    cfg: BudgetConfig = CFG,
) -> BuyerFinancials:
    use_hdb_loan = monthly_income <= cfg.hdb_income_ceiling

    cpf_oa = estimate_cpf_oa_balance(current_age, monthly_income, cfg)
    cash = estimate_cash_savings(current_age, monthly_income, cfg)
    max_mortgage, debt_service_benchmark = compute_max_monthly_mortgage(monthly_income, cfg)
    loan, tenure = compute_max_loan_quantum(max_mortgage, use_hdb_loan, current_age, cfg)
    max_val, max_cov = allocate_resale_cash(
        loan_quantum=loan,
        cpf_oa_balance=cpf_oa,
        cash_savings=cash,
        use_hdb_loan=use_hdb_loan,
        forced_max_cov=forced_max_cov,
        cfg=cfg,
    )

    return BuyerFinancials(
        max_valuation=round(max_val, -3),
        max_cov=round(max_cov, -3),
        effective_ceiling=round(max_val + max_cov, -3),
        cpf_oa_balance=round(cpf_oa, 2),
        cash_savings=round(cash, 2),
        loan_quantum=round(loan, 2),
        loan_type="hdb" if use_hdb_loan else "bank",
        monthly_mortgage_capacity=round(max_mortgage, 2),
        effective_tenure_years=tenure,
        monthly_income=monthly_income,
        total_debt_service_benchmark_monthly=round(debt_service_benchmark, 2),
    )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_financials_for_pool(
    buyers: list[dict],
    income_bands: dict[str, dict[str, float | None]] = INCOME_BANDS,
    cfg: BudgetConfig = CFG,
) -> list[dict]:
    """
    Adds 'financials' to each buyer dict.

    Each buyer dict must contain:
        - age
        - income_band

    Logic:
        monthly_income = upper bound of buyer's own band
        max_cov       = upper bound of buyer's own band
    """
    for buyer in buyers:
        band_upper = resolve_income_band_upper(
            income_band=buyer["income_band"],
            income_bands=income_bands,
        )

        financials = compute_buyer_financials(
            current_age=buyer["age"],
            monthly_income=band_upper,
            forced_max_cov=band_upper,
            cfg=cfg,
        )

        buyer["financials"] = financials.__dict__

    return buyers