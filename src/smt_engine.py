
from z3 import *
from policy import Borrower


def verify_with_z3(b: Borrower):

    solver = Solver()

    # 1. Declare financial variables
    I = Real('I')
    E = Real('E')
    D = Real('D')
    S = Real('S')
    P = Real('P')
    C = Real('C')
    T = Real('T')

    Age = Int('Age')
    Rate = Real('Rate')

    # 2. Assign borrower values
    solver.add(I == RealVal(str(b.monthly_income)))
    solver.add(E == RealVal(str(b.family_expenses)))
    solver.add(D == RealVal(str(b.existing_emi)))
    solver.add(S == RealVal(str(b.monthly_savings)))
    solver.add(P == RealVal(str(b.proposed_emi)))
    solver.add(C == RealVal(str(b.bank_credits)))
    solver.add(T == RealVal(str(b.turnover)))

    solver.add(Age == b.person_age)
    solver.add(Rate == RealVal(str(b.loan_int_rate)))

    # 3. Policy 1: Repayment surplus
    surplus = I - E - D - S

    solver.assert_and_track(
        And(P >= 0, P <= surplus),
        "repayment_surplus"
    )

    # 4. Policy 2: Business profitability
    solver.assert_and_track(
        BoolVal(b.profitable_fy1 and b.profitable_fy2),
        "business_profitability"
    )

    # 5. Policy 3: Bank credit verification
    solver.assert_and_track(
        C >= T,
        "bank_credit_verification"
    )

    # 6. Policy 4: Repayment history
    repayment_history = Or(
        b.repayment_bounces == 0,
        And(
            b.repayment_bounces > 0,
            b.repayment_bounces <= 2,
            b.bounces_cleared_within_3_days,
            b.deviation_approved
        )
    )

    solver.assert_and_track(
        BoolVal(repayment_history),
        "repayment_history"
    )

    # 7. Policy 5: Age eligibility
    solver.assert_and_track(
        And(Age >= 18, Age <= 100),
        "age_eligibility"
    )

    # 8. Policy 6: Interest rate verification
    solver.assert_and_track(
        And(Rate >= 5.0, Rate <= 25.0),
        "interest_rate"
    )

    # 9. Evaluate constraints
    result = solver.check()

    if result == sat:
        return "SAT", "Compliant"

    elif result == unsat:

        unsat_core = [
            str(constraint)
            for constraint in solver.unsat_core()
        ]

        return (
            "UNSAT",
            {
                "message": "Borrower violates lending policies",
                "violated_constraints": unsat_core
            }
        )

    else:
        return "UNKNOWN", solver.reason_unknown()
