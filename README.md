# LendVerify: AI-Powered Loan Verification System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/frontend-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Backend](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Logic Engine](https://img.shields.io/badge/SMT-Z3%20Theorem%20Prover-orange.svg)](https://github.com/Z3Prover/z3)

**LendVerify** is a hybrid loan verification platform combining statistical Machine Learning models with Satisfiability Modulo Theories (SMT) formal verification via the Z3 Theorem Prover. It provides evidence-led loan screening, model performance comparison, and deterministic policy compliance checks.

---

## Key Features

### 1. Hybrid Decision Engine
- **ML Predictive Models:** Evaluates borrower risk profiles using side-by-side performance tracking:
  - **Model A (Random Forest):** $94.6\%$ Accuracy ($F_1$ Score: 0.944)
  - **Model B (XGBoost):** $94.4\%$ Accuracy ($F_1$ Score: 0.941)
- **Z3 SMT Policy Verification:** Executes mathematical proof-based verification on loan rules to ensure deterministic compliance ($62.6\%$ compliance rate across verified dataset).

### 2. Formally Verified Policy Rules Engine
Evaluates borrower data against six strict verification predicates:
- `adult_eligibility`: Age and legal borrower status criteria.
- `positive_income_and_loan`: Validation of non-negative cash flows and principal requested.
- `income_ratio_limit`: Debt-to-income and proposed EMI safety thresholds.
- `valid_interest_rate`: Interest rate bounds compliance.
- `employment_history_bounds`: Minimum required employment stability checks.
- `credit_history_bounds`: Hard risk limits on past repayment behavior.

### 3. Interactive Logic Lab & Borrower Assessment
- Dynamic financial evaluation tool calculating real-time feasibility across key metrics:
  - Financial year profitability check (FY1 & FY2)
  - Annual turnover ($\text{₹}$) and bank credits ($\text{₹}$)
  - Repayment bounce analysis & 3-day resolution tracking
  - Monthly disposable income, existing EMIs, savings, and family expenses
  - Proposed EMI calculation vs. net capacity
- Single-click automated policy assessment engine.

### 4. Enterprise Audit & Reporting
- Batch verification engine supporting dataset-wide rule validation ($32,581+$ rows evaluated).
- Granular borrower-level breakdown isolating **PASS**, **FAIL**, and **UNVERIFIABLE** policy statuses.
- Automated generation of downloadable Z3 Audit Reports (`.txt`) for regulatory compliance logging.

---

## Tech Stack

- **Frontend:** Streamlit
- **Backend API:** FastAPI
- **Formal Verification Engine:** Z3 Theorem Prover (Python SDK)
- **Machine Learning Engine:** Scikit-Learn, XGBoost
- **Data Analysis & Visualization:** Pandas, NumPy, Plotly

---

## Architecture Overview

```
 ┌─────────────────────────────────────────────────────────┐
 │                   Streamlit UI (App)                    │
 └──────────────┬────────────────────────────▲─────────────┘
                │                            │
                │ HTTP / REST API             │ JSON Response
                ▼                            │
 ┌─────────────────────────────────────────────────────────┐
 │                      FastAPI Backend                    │
 ├────────────────────────────┬────────────────────────────┤
 │    ML Inference Pipeline   │     Z3 SMT Logic Engine    │
 │ (Random Forest / XGBoost)  │   (Policy Proof Checker)   │
 └────────────────────────────┴────────────────────────────┘
```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 1. Repository Setup

```bash
git clone https://github.com/your-username/lendverify.git
cd lendverify
```

### 2. Environment Setup

Create and activate a virtual environment:

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Step 1: Start the FastAPI Backend
Launch the FastAPI server first to serve prediction and verification endpoints:

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend server will run on `http://127.0.0.1:8000`.

### Step 2: Start the Streamlit Frontend
In a separate terminal window (with the virtual environment activated), start the interface:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to access **LendVerify**.

---

## Project Structure

```
lendverify/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── z3_engine.py         # Z3 SMT logic formulas & rule proofs
│   └── ml_models.py         # Random Forest & XGBoost model pipelines
├── app.py                   # Streamlit frontend app
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.