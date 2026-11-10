
from dataclasses import dataclass


@dataclass
class Borrower:
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_status: int
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float


def check_policies(b: Borrower) -> dict:

    rules = {

        # Rule 1: Adult eligibility
        "Adult eligibility": (
            18 <= b.person_age <= 100
        ),

        # Rule 2: Valid employment history
        "Valid employment history": (
            0 <= b.person_emp_length <= 60
        ),

        # Rule 3: Valid annual income
        "Positive annual income": (
            b.person_income > 0
        ),

        # Rule 4: Loan amount must be positive
        "Valid loan amount": (
            b.loan_amnt > 0
        ),

        # Rule 5: Interest rate bounds
        "Valid interest rate": (
            5.0 <= b.loan_int_rate <= 25.0
        ),

        # Rule 6: Loan-to-income ratio
        "Loan income ratio limit": (
            0 < b.loan_percent_income <= 0.50
        )
    }

    return {
        "rules": rules,
        "compliant": all(rules.values())
    }
