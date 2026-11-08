from z3 import *

def verify_borrower_z3(monthly_income, family_expenses, existing_emi, monthly_savings, proposed_emi, bank_credits, turnover, bounces, cleared, deviation):
    # Create Z3 Real variables
    I = Real('I')
    E = Real('E')
    D = Real('D')
    S = Real('S')
    P = Real('P')
    C = Real('C')
    T = Real('T')
    
    solver = Solver()
    
    # Add borrower values to the solver
    solver.add(I == monthly_income)
    solver.add(E == family_expenses)
    solver.add(D == existing_emi)
    solver.add(S == monthly_savings)
    solver.add(P == proposed_emi)
    solver.add(C == bank_credits)
    solver.add(T == turnover)
    
    # Add Policy Constraints
    # Rule 1: Proposed EMI must be positive
    solver.add(P >= 0)
    
    # Rule 2: EMI cannot exceed available surplus
    solver.add(P <= (I - E - D - S))
    
    # Rule 3: Bank credits must cover turnover
    solver.add(C >= T)
    
    # Rule 4: Repayment history
    if bounces > 2 or (bounces > 0 and not (cleared and deviation)):
        # If history fails, we add a false constraint to force UNSAT
        solver.add(False)
    
    # Check satisfiability
    result = solver.check()
    
    if result == sat:
        return "SAT", "Compliant"
    else:
        return "UNSAT", "Non-Compliant"