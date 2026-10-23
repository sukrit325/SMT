from dataclasses import dataclass

@dataclass
class Borrower:
    profitable_fy1: bool
    profitable_fy2: bool
    bank_credits: float
    turnover: float
    repayment_bounces: int
    bounces_cleared_within_3_days: bool
    deviation_approved: bool
    monthly_income: float
    existing_emi: float
    monthly_savings: float
    family_expenses: float
    proposed_emi: float
    
def check_policies(b: Borrower) -> dict:
    rules = {
        "Two profitable financial years" :(
            b.profitable_fy1 and b.profitable_fy2
        ),
        "Bank credits at least equal turnover" : (
            b.bank_credits >= b.turnover
        ),
        "Repayment history" : (
            b.repayment_bounces == 0
            or (
                0 < b.repayment_bounces <= 2
                and b.bounces_cleared_within_3_days
                and b.deviation_approved
            )
        ),
    }
    available_surplus = (
        b.monthly_income - b.family_expenses - b.existing_emi - b.monthly_savings
    )
    rules["Proposed EMI within available surplus"] = (
        b.proposed_emi >= 0
        and b.proposed_emi <= available_surplus
    )
    return {
        "rules": rules,
        "available_surplus": available_surplus,
        "compliant": all(rules.values()),
    }
