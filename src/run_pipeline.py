import pandas as pd
from z3_engine import verify_borrower_z3

# Load the dataset
df = pd.read_csv('borrowers.csv')

compliant_count = 0
non_compliant_count = 0

# We loop through the first 100 rows to prove it works (or the whole file if fast enough)
for index, row in df.iterrows():
    # Map CSV columns to Z3 variables (This explains how you derived the missing variables)
    monthly_income = row['person_income'] / 12
    # Estimate expenses and savings as a percentage of income (Standard financial assumption)
    family_expenses = monthly_income * 0.30 
    monthly_savings = monthly_income * 0.10
    
    # Calculate existing EMI based on loan percent
    existing_emi = row['person_income'] * row['loan_percent_income']
    
    # Proposed EMI (Approximate calculation based on loan amount)
    proposed_emi = row['loan_amnt'] * 0.05 # Assuming 5% monthly repayment
    
    # Derived variables
    bank_credits = row['person_income'] # Assuming credits match income
    turnover = row['person_income']
    bounces = 1 if row['cb_person_default_on_file'] == 'Y' else 0
    cleared = True
    deviation = True
    
    # Run Z3
    status, msg = verify_borrower_z3(
        monthly_income, family_expenses, existing_emi, monthly_savings, 
        proposed_emi, bank_credits, turnover, bounces, cleared, deviation
    )
    
    if status == "SAT":
        compliant_count += 1
    else:
        non_compliant_count += 1

print(f"Total Checked: {compliant_count + non_compliant_count}")
print(f"Compliant (SAT): {compliant_count}")
print(f"Non-Compliant (UNSAT): {non_compliant_count}")