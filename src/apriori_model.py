# src/apriori_model.py
"""
Create a transaction-item matrix and run Apriori + association rules (mlxtend).
Save frequent itemsets and rules to a pickle or CSV.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
import joblib

FEATURE_PATH = Path("data/processed/featurized_sales_data.csv")
RULES_PATH = Path("models/apriori_rules.pkl")
FREQUENT_CSV = Path("data/processed/frequent_itemsets.csv")
RULES_CSV = Path("data/processed/association_rules.csv")

def create_transaction_matrix(df: pd.DataFrame):
    # Columns: transaction_id, product_id, quantity_sold
    txn = df.groupby(["transaction_id","product_id"])["quantity_sold"].sum().unstack(fill_value=0)
    # convert quantities to 1/0 (presence)
    txn = (txn > 0).astype(int)
    return txn

def run_apriori(txn_matrix: pd.DataFrame, min_support=0.01, min_confidence=0.2):
    frequent_itemsets = apriori(txn_matrix, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return frequent_itemsets, rules

def run(min_support=0.01, min_confidence=0.2):
    df = pd.read_csv(FEATURE_PATH)
    txn = create_transaction_matrix(df)
    frequent_itemsets, rules = run_apriori(txn, min_support=min_support, min_confidence=min_confidence)
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rules, RULES_PATH)
    frequent_itemsets.to_csv(FREQUENT_CSV, index=False)
    rules.to_csv(RULES_CSV, index=False)
    print(f"Saved rules to {RULES_PATH}, CSVs to {FREQUENT_CSV} and {RULES_CSV}")
    return frequent_itemsets, rules

if __name__ == "__main__":
    run()
