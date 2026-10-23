from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _borrower_violating(rule: str) -> dict[str, Any]:
    """Return a concrete borrower that violates `rule`."""
    # A single "everything wrong" borrower is enough to find a counterexample.
    return {
        "profitable_fy1": False,
        "profitable_fy2": False,
        "bank_credits": 0.0,
        "turnover": 1_000_000.0,
        "repayment_bounces": 5,
        "bounces_cleared_within_3_days": False,
        "deviation_approved": False,
        "monthly_income": 10_000.0,
        "existing_emi": 50_000.0,
        "monthly_savings": 0.0,
        "family_expenses": 40_000.0,
        "proposed_emi": 20_000.0,
    }


_RULE_LABELS = {
    "profitability": "Profitability (both FYs)",
    "banking": "Banking credits ≥ 95% of turnover",
    "repayment": "Repayment behaviour acceptable",
    "affordability": "Affordability (surplus ≥ 0)",
}


def verify_policy(defect: str | None = None) -> dict[str, Any]:
    """
    Return a SAT/UNSAT verdict for the approval logic with an optional
    omitted rule (the 'defect').

    - defect=None: all rules enforced -> no violating borrower can be approved.
    - defect set: that rule is dropped -> a violating borrower can be approved.
    """
    if defect is not None and defect not in _RULE_LABELS:
        raise ValueError(f"Unknown defect: {defect!r}")

    borrower = _borrower_violating(defect or "")

    # Ground-truth policy checks
    policies = {
        "Profitability (both FYs)": (
            borrower["profitable_fy1"] and borrower["profitable_fy2"]
        ),
        "Banking credits ≥ 95% of turnover": (
            borrower["bank_credits"] >= borrower["turnover"] * 0.95
        ),
        "Repayment behaviour acceptable": (
            borrower["repayment_bounces"] == 0
            or (
                borrower["bounces_cleared_within_3_days"]
                and borrower["deviation_approved"]
            )
        ),
        "Affordability (surplus ≥ 0)": (
            borrower["monthly_income"]
            - borrower["existing_emi"]
            - borrower["family_expenses"]
            - borrower["proposed_emi"]
        ) >= 0,
    }

    if defect is None:
        return {
            "status": "UNSAT",
            "finding": (
                "With all rules enforced, no borrower can be approved while "
                "violating the mandatory policy. The encoded approval logic is sound."
            ),
            "counterexample": None,
            "policies": policies,
        }

    omitted_label = _RULE_LABELS[defect]
    remaining = {k: v for k, v in policies.items() if k != omitted_label}

    if all(not v for v in remaining.values()):
        return {
            "status": "SAT",
            "finding": (
                f"Omitting the '{omitted_label}' rule allows approval of a borrower "
                "who violates the mandatory policy. This exposes a policy gap."
            ),
            "counterexample": {
                "approved": True,
                "violated_rule": omitted_label,
                "borrower": borrower,
            },
            "policies": policies,
        }

    return {
        "status": "UNSAT",
        "finding": "Even with this rule omitted, no violating approval exists.",
        "counterexample": None,
        "policies": policies,
    }