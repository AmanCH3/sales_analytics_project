"""
Enhanced Apriori Engine with Category-First Recommendations:
- Builds transaction-item matrix
- Extracts frequent itemsets + association rules
- Generates product-level + category-level hierarchical recommendations
- Suggests aisle + rack placement
- Includes data quality checks, metrics, and error handling
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from pathlib import Path
import joblib
import warnings
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict

# -------------------------------------------------------
# Setup Logging
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
FEATURE_PATH = Path("data/raw/nepal_electronics_transactions_6000_with_names.csv")
OUT_DIR = Path("data/processed")
MODEL_DIR = Path("models")

CACHE_PATH = MODEL_DIR / "txn_matrix_cache.pkl"
RULES_PATH = MODEL_DIR / "apriori_rules.pkl"

FREQUENT_CSV = OUT_DIR / "frequent_itemsets.csv"
RULES_CSV = OUT_DIR / "association_rules.csv"
PRODUCT_RECO_CSV = OUT_DIR / "product_category_recommendations.csv"
CATEGORY_SUMMARY_CSV = OUT_DIR / "category_summary.csv"
PLACEMENT_CSV = OUT_DIR / "rack_aisle_suggestions.csv"
METRICS_CSV = OUT_DIR / "apriori_metrics.csv"
TEMPORAL_CSV = OUT_DIR / "temporal_patterns.csv"

# -------------------------------------------------------
# 1. Data Quality & Validation
# -------------------------------------------------------
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Validating input data...")
    required_cols = ['transaction_id', 'product_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    df = df.dropna(subset=['transaction_id', 'product_name'])
    df = df[df['product_name'].str.strip() != '']
    if len(df) < 10:
        raise ValueError("Insufficient data: need at least 10 transactions")
    logger.info(f"✓ Data validation complete: {len(df)} rows validated")
    return df

def print_data_quality_report(df: pd.DataFrame):
    print("\n" + "="*70)
    print("📊 DATA QUALITY REPORT")
    print("="*70)
    print(f"Total rows: {len(df):,}")
    print(f"Unique transactions: {df['transaction_id'].nunique():,}")
    print(f"Unique products: {df['product_name'].nunique():,}")
    items_per_txn = df.groupby('transaction_id').size()
    print(f"Avg items per transaction: {items_per_txn.mean():.2f}")
    print(f"Min items per transaction: {items_per_txn.min()}")
    print(f"Max items per transaction: {items_per_txn.max()}")
    print(f"Median items per transaction: {items_per_txn.median():.0f}")
    product_freq = df['product_name'].value_counts()
    print(f"Most common product: {product_freq.index[0]} ({product_freq.iloc[0]} times)")
    print(f"Products appearing once: {len(product_freq[product_freq==1])}")
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        print(f"Date Range: {df['transaction_date'].min()} → {df['transaction_date'].max()}")
    optional_cols = ['aisle_id', 'rack_id', 'category', 'price']
    available_cols = [col for col in optional_cols if col in df.columns]
    if available_cols:
        print(f"Available optional columns: {', '.join(available_cols)}")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    print("="*70 + "\n")

# -------------------------------------------------------
# 2. Build Transaction Matrix
# -------------------------------------------------------
def create_transaction_matrix(df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        logger.info("Loading cached transaction matrix...")
        try:
            return joblib.load(CACHE_PATH)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Rebuilding matrix...")
    logger.info("Building transaction matrix...")
    df["qty"] = 1
    txn = df.pivot_table(index="transaction_id", columns="product_name", values="qty", aggfunc="sum", fill_value=0)
    txn = txn.astype(bool)
    if use_cache:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(txn, CACHE_PATH)
        logger.info("✓ Transaction matrix cached")
    return txn

# -------------------------------------------------------
# 3. Run Apriori + Association Rules
# -------------------------------------------------------
def run_apriori(txn_matrix: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.2, max_len: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Running Apriori (support={min_support}, confidence={min_confidence})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        frequent_itemsets = apriori(txn_matrix, min_support=min_support, max_len=max_len, use_colnames=True)
    if len(frequent_itemsets) == 0:
        logger.warning("No frequent itemsets found.")
        return pd.DataFrame(), pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    logger.info(f"✓ Found {len(frequent_itemsets)} itemsets, {len(rules)} rules")
    return frequent_itemsets, rules

# -------------------------------------------------------
# 4. Metrics
# -------------------------------------------------------
def calculate_metrics(rules: pd.DataFrame, frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
    if len(rules) == 0:
        return pd.DataFrame([{"timestamp": datetime.now().isoformat(), "total_rules": 0, "frequent_itemsets": len(frequent_itemsets), "message": "No rules"}])
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_rules": len(rules),
        "frequent_itemsets_count": len(frequent_itemsets),
        "avg_confidence": rules["confidence"].mean(),
        "max_confidence": rules["confidence"].max(),
        "min_confidence": rules["confidence"].min(),
        "avg_lift": rules["lift"].mean(),
        "max_lift": rules["lift"].max(),
        "min_lift": rules["lift"].min(),
        "avg_support": rules["support"].mean(),
        "rules_with_lift_gt_1": len(rules[rules["lift"] > 1.0]),
        "rules_with_lift_gt_1_5": len(rules[rules["lift"] > 1.5]),
        "rules_with_lift_gt_2": len(rules[rules["lift"] > 2.0]),
        "strong_rules_pct": len(rules[rules["lift"] > 1.5]) / len(rules) * 100
    }
    return pd.DataFrame([metrics])

# -------------------------------------------------------
# 5. Hierarchical Product + Category Recommendations
# -------------------------------------------------------
def generate_hierarchical_recommendations(df: pd.DataFrame, rules: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        - category_summary: avg lift etc per category→category
        - product_category_reco: detailed product→product + category info
    """
    if 'category' not in df.columns or len(rules) == 0:
        return pd.DataFrame(), pd.DataFrame()
    prod_cat_map = df.groupby('product_name')['category'].first().to_dict()
    reco = []
    cat_summary = {}
    for _, row in rules.iterrows():
        ant_list = list(row['antecedents'])
        cons_list = list(row['consequents'])
        for p_from in ant_list:
            for p_to in cons_list:
                cat_from = prod_cat_map.get(p_from)
                cat_to = prod_cat_map.get(p_to)
                if not cat_from or not cat_to:
                    continue
                type_assoc = 'same_category' if cat_from == cat_to else 'cross_category'
                reco.append({
                    "product": p_from,
                    "recommended_product": p_to,
                    "product_category": cat_from,
                    "recommended_category": cat_to,
                    "type": type_assoc,
                    "lift": row['lift'],
                    "confidence": row['confidence'],
                    "support": row['support']
                })
                # Category summary
                key = (cat_from, cat_to)
                if key not in cat_summary:
                    cat_summary[key] = []
                cat_summary[key].append(row['lift'])
    product_category_reco = pd.DataFrame(reco)
    # Category summary
    summary_rows = []
    for (cat_from, cat_to), lifts in cat_summary.items():
        avg_lift = np.mean(lifts)
        summary_rows.append({
            "category_from": cat_from,
            "category_to": cat_to,
            "num_product_pairs": len(lifts),
            "avg_lift": avg_lift,
            "strength": 'Very Strong' if avg_lift>3 else 'Strong' if avg_lift>2 else 'Moderate' if avg_lift>1.5 else 'Weak'
        })
    category_summary = pd.DataFrame(summary_rows).sort_values('avg_lift', ascending=False)
    return category_summary, product_category_reco

# -------------------------------------------------------
# 6. Placement Suggestions
# -------------------------------------------------------
def generate_placement_recommendations(df: pd.DataFrame, reco_df: pd.DataFrame) -> pd.DataFrame:
    if len(reco_df) == 0 or 'aisle_id' not in df.columns or 'rack_id' not in df.columns:
        return pd.DataFrame()
    prod_map = df.groupby("product_name")[["aisle_id","rack_id"]].first().to_dict("index")
    placement = []
    for _, row in reco_df.iterrows():
        p1 = row["product"]
        p2 = row["recommended_product"]
        if p1 in prod_map and p2 in prod_map:
            p1_info = prod_map[p1]; p2_info = prod_map[p2]
            same_aisle = p1_info["aisle_id"]==p2_info["aisle_id"]
            same_rack = p1_info["rack_id"]==p2_info["rack_id"]
            if same_rack:
                suggestion=f"✓ Already together: {p1} and {p2} on same rack"
            elif same_aisle:
                suggestion=f"⚠ Same aisle, different racks: Consider moving {p1} closer to {p2}"
            else:
                suggestion=f"❗ Different aisles: Consider relocating {p1} near {p2}"
            placement.append({
                "product_1": p1,
                "product_2": p2,
                "p1_aisle": p1_info["aisle_id"],
                "p1_rack": p1_info["rack_id"],
                "p2_aisle": p2_info["aisle_id"],
                "p2_rack": p2_info["rack_id"],
                "same_aisle": same_aisle,
                "same_rack": same_rack,
                "suggestion": suggestion,
                "lift": row["lift"],
                "confidence": row["confidence"],
                "priority": "High" if row["lift"]>2 else "Medium" if row["lift"]>1.5 else "Low"
            })
    return pd.DataFrame(placement).sort_values("lift", ascending=False)

# -------------------------------------------------------
# 7. Temporal Patterns
# -------------------------------------------------------
def analyze_temporal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if 'transaction_date' not in df.columns:
        return pd.DataFrame()
    df['transaction_date']=pd.to_datetime(df['transaction_date'], errors='coerce')
    df=df.dropna(subset=['transaction_date'])
    if len(df)==0: return pd.DataFrame()
    df['year_month']=df['transaction_date'].dt.to_period('M').astype(str)
    df['day_of_week']=df['transaction_date'].dt.day_name()
    monthly=df.groupby(['year_month','product_name']).size().reset_index(name='count')
    monthly['period_type']='monthly'
    dow=df.groupby(['day_of_week','product_name']).size().reset_index(name='count')
    dow['period_type']='day_of_week'; dow=dow.rename(columns={'day_of_week':'year_month'})
    return pd.concat([monthly,dow],ignore_index=True)

# -------------------------------------------------------
# 8. Run Full Pipeline
# -------------------------------------------------------
def run(min_support=0.005,min_confidence=0.05,max_len=None,use_cache=True,generate_temporal=True):
    try:
        logger.info("="*70)
        logger.info("🚀 STARTING ENHANCED APRIORI PIPELINE (Category-First)")
        logger.info("="*70)
        df=pd.read_csv(FEATURE_PATH)
        df=validate_data(df)
        print_data_quality_report(df)
        txn=create_transaction_matrix(df,use_cache)
        frequent_itemsets,rules=run_apriori(txn,min_support,min_confidence,max_len)
        metrics_df=calculate_metrics(rules,frequent_itemsets)
        OUT_DIR.mkdir(parents=True,exist_ok=True); MODEL_DIR.mkdir(parents=True,exist_ok=True)
        frequent_itemsets.to_csv(FREQUENT_CSV,index=False)
        rules.to_csv(RULES_CSV,index=False)
        metrics_df.to_csv(METRICS_CSV,index=False)
        joblib.dump(rules,RULES_PATH)
        category_summary, product_category_reco=generate_hierarchical_recommendations(df,rules)
        category_summary.to_csv(CATEGORY_SUMMARY_CSV,index=False)
        product_category_reco.to_csv(PRODUCT_RECO_CSV,index=False)
        placement_df=generate_placement_recommendations(df,product_category_reco)
        placement_df.to_csv(PLACEMENT_CSV,index=False)
        temporal_df=analyze_temporal_patterns(df) if generate_temporal else pd.DataFrame()
        if generate_temporal and len(temporal_df)>0:
            temporal_df.to_csv(TEMPORAL_CSV,index=False)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        return {
            'frequent_itemsets': frequent_itemsets,
            'rules': rules,
            'category_summary': category_summary,
            'product_category_reco': product_category_reco,
            'placement': placement_df,
            'metrics': metrics_df,
            'temporal_patterns': temporal_df
        }
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

# -------------------------------------------------------
# 9. Run script
# -------------------------------------------------------
if __name__=="__main__":
    results=run(min_support=0.005,min_confidence=0.05,use_cache=True,generate_temporal=True)
    print("\n🎉 Enhanced analysis complete! Check the 'data/processed' folder for results.")
