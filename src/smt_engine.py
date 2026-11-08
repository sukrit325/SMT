from z3 import *
from policy import Borrower

def verify_with_z3(b: Borrower):
    # 1. Declare Z3 variables
    I = Real('I'); E = Real('E'); D = Real('D'); S = Real('S')
    P = Real('P'); C = Real('C'); T = Real('T')
    Age = Int('Age'); Rate = Real('Rate')
    
    # 2. Assign values from the Borrower dataclass
    solver = Solver()
    solver.add(I == b.monthly_income)
    solver.add(E == b.family_expenses)
    solver.add(D == b.existing_emi)
    solver.add(S == b.monthly_savings)
    solver.add(P == b.proposed_emi)
    solver.add(C == b.bank_credits)
    solver.add(T == b.turnover)
    solver.add(Age == b.person_age)
    solver.add(Rate == b.loan_int_rate)
    
    # 3. Add the 6 constraints (matching policy.py)
    solver.add(P >= 0)
    solver.add(P <= (I - E - D - S)) # Surplus rule
    solver.add(C >= T)               # Credits rule
    solver.add(Age >= 18)            # Adult
    solver.add(Age <= 100)           # Age Outlier limit
    solver.add(Rate >= 5.0)          # Rate lower bound
    solver.add(Rate <= 25.0)         # Rate upper bound
    
    # Hardcode boolean rules (since Z3 doesn't do Booleans easily here)
    if not (b.profitable_fy1 and b.profitable_fy2):
        solver.add(False)
    if b.repayment_bounces > 0 and not (b.bounces_cleared_within_3_days and b.deviation_approved):
        solver.add(False)
        
    # 4. Check and return UNSAT Core
    if solver.check() == sat:
        return "SAT", "Compliant"
    else:
        # This proves you understand Z3: return the exact failed constraint
        unsat_core = solver.unsat_core()
        return "UNSAT", f"Violated Constraints: {unsat_core}"