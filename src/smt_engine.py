
import math
from z3 import *
from policy import Borrower


class LendVerifyEngine:

    def __init__(self):
        self.solver = Solver()

        # Declare variables only once
        self.Age = Int("Age")
        self.Income = Real("Income")
        self.Employment = Real("Employment")
        self.LoanAmount = Real("LoanAmount")
        self.Rate = Real("Rate")
        self.LoanRatio = Real("LoanRatio")

    def verify(self, b: Borrower):

        values = [
            b.person_age,
            b.person_income,
            b.person_emp_length,
            b.loan_amnt,
            b.loan_int_rate,
            b.loan_percent_income
        ]

        # Reject missing or invalid values
        try:
            if any(
                v is None or not math.isfinite(float(v))
                for v in values
            ):
                return "UNKNOWN", "Missing or invalid input"

            if float(b.person_age) != int(b.person_age):
                return "UNKNOWN", "Invalid age"

        except (TypeError, ValueError):
            return "UNKNOWN", "Invalid input"

        s = self.solver

        # Isolate each borrower's constraints
        s.push()

        try:
            # Assign current borrower values
            s.add(self.Age == IntVal(int(b.person_age)))
            s.add(self.Income == RealVal(str(float(b.person_income))))
            s.add(self.Employment == RealVal(str(float(b.person_emp_length))))
            s.add(self.LoanAmount == RealVal(str(float(b.loan_amnt))))
            s.add(self.Rate == RealVal(str(float(b.loan_int_rate))))
            s.add(self.LoanRatio == RealVal(str(float(b.loan_percent_income))))

            # Six tracked policies
            policies = [
                (And(self.Age >= 18, self.Age <= 100),
                 "age_eligibility"),

                (And(self.Employment >= 0, self.Employment <= 60),
                 "employment_history_bounds"),

                (self.Income > 0,
                 "positive_annual_income"),

                (self.LoanAmount > 0,
                 "positive_loan_amount"),

                (And(self.Rate >= 5, self.Rate <= 25),
                 "interest_rate_bounds"),

                (And(self.LoanRatio > 0, self.LoanRatio <= 0.50),
                 "loan_income_ratio_limit")
            ]

            for condition, name in policies:
                s.assert_and_track(condition, Bool(name))

            result = s.check()

            if result == sat:
                return "SAT", "All six policies satisfied"

            if result == unsat:
                core = [str(x) for x in s.unsat_core()]

                return "UNSAT", {
                    "message": "Policy violations detected",
                    "violated_constraints": core
                }

            return "UNKNOWN", s.reason_unknown()

        finally:
            # Remove current borrower's constraints
            s.pop()
