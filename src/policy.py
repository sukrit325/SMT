from dataclasses import dataclass

@dataclass
class Borrower:
    # Custom business rules (4)
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
    # Standard LendingClub rules (2)
    person_age: int
    loan_int_rate: float
    
def check_policies(b: Borrower) -> dict:
    rules = {
        # Rule 1
        "Two profitable financial years": (
            b.profitable_fy1 and b.profitable_fy2
        ),
        # Rule 2
        "Bank credits at least equal turnover": (
            b.bank_credits >= b.turnover
        ),
        # Rule 3
        "Repayment history": (
            b.repayment_bounces == 0
            or (
                0 < b.repayment_bounces <= 2
                and b.bounces_cleared_within_3_days
                and b.deviation_approved
            )
        ),
        # Rule 4
        "Adult eligibility": (
            b.person_age >= 18 and b.person_age <= 100
        ),
        # Rule 5
        "Valid interest rate": (
            b.loan_int_rate >= 5.0 and b.loan_int_rate <= 25.0
        )
    }
    
    available_surplus = (
        b.monthly_income - b.family_expenses - b.existing_emi - b.monthly_savings
    )
    
    # Rule 6
    rules["Proposed EMI within available surplus"] = (
        b.proposed_emi >= 0
        and b.proposed_emi <= available_surplus
    )
    
    return {
        "rules": rules,
        "available_surplus": available_surplus,
        "compliant": all(rules.values()),
    }