import streamlit as st
import pandas as pd
from io import BytesIO
from typing import List, Optional

# --- Node & Hierarchy Utilities ---
class Node:
    def __init__(self, name, target, constraint=None, children: Optional[List['Node']] = None):
        self.name = name
        self.target = float(target)
        self.constraint = float(constraint) if constraint is not None else 0.0
        self.allocation = 0.0
        self.children = children or []
    def __repr__(self):
        return f"Node({self.name}, {self.target}, {self.constraint}, alloc={self.allocation})"

def build_hierarchy(df: pd.DataFrame) -> List[Node]:
    sec = {
        row['Ticker']: {
            'risk': row['risk'],
            'asset_class': row['asset_class'],
            'target': row['target'],
            'constrained': row['constrained']
        }
        for _, row in df.iterrows()
    }
    # Group by risk and asset_class
    risk_map = {}
    for ticker, meta in sec.items():
        risk_map.setdefault(meta['risk'], {})
        risk_map[meta['risk']].setdefault(meta['asset_class'], []).append(ticker)
    roots: List[Node] = []
    for risk, acs in risk_map.items():
        ac_nodes: List[Node] = []
        for ac, tickers in acs.items():
            tgt = sum(sec[t]['target'] for t in tickers)
            con = sum(sec[t]['constrained'] or 0 for t in tickers)
            children = [Node(t, sec[t]['target'], sec[t]['constrained']) for t in tickers]
            ac_nodes.append(Node(ac, tgt, con, children))
        roots.append(Node(risk,
                          sum(n.target for n in ac_nodes),
                          sum(n.constraint for n in ac_nodes),
                          ac_nodes))
    return roots

def proportional_cascading_overshoot(nodes: List[Node], overshoot: float):
    eligible = [n for n in nodes if n.constraint < n.target]
    while overshoot > 1e-6 and eligible:
        sum_targets = sum(n.target for n in eligible)
        next_eligible = []
        redistributed = 0.0
        for n in eligible:
            reduction = overshoot * (n.target / sum_targets) if sum_targets > 0 else 0
            new_alloc = n.allocation - reduction
            if new_alloc < n.constraint:
                redistributed += n.constraint - new_alloc
                n.allocation = n.constraint
            else:
                n.allocation = new_alloc
                next_eligible.append(n)
        overshoot = redistributed
        eligible = next_eligible

def improved_waterfall_rebalance(nodes: List[Node]) -> None:
    total_target     = sum(n.target     for n in nodes)
    total_constraint = sum(n.constraint for n in nodes)
    extra = total_target - total_constraint
    overshoot_total = sum(max(n.constraint - n.target, 0.0) for n in nodes)
    if overshoot_total > 0 and extra > 0:
        for n in nodes:
            if n.constraint > n.target:
                n.allocation = n.constraint
            else:
                n.allocation = n.target
        proportional_cascading_overshoot(nodes, overshoot_total)
    elif extra > 0:
        head_nodes = [n for n in nodes if n.constraint < n.target]
        head_sum   = sum(n.target - n.constraint for n in head_nodes)
        for n in nodes:
            if n.constraint >= n.target:
                n.allocation = n.constraint
            else:
                n.allocation = n.constraint + ((n.target - n.constraint) * (extra / head_sum) if head_sum else 0)
    else:
        for n in nodes:
            n.allocation = n.constraint
    for parent in nodes:
        if parent.children:
            child_total = sum(c.target for c in parent.children) or 1.0
            for c in parent.children:
                c.target = (c.target / child_total) * parent.allocation if child_total else 0
            improved_waterfall_rebalance(parent.children)

def collect_leaf_allocations(nodes, results=None):
    if results is None:
        results = {}
    for n in nodes:
        if n.children:
            collect_leaf_allocations(n.children, results)
        else:
            results[n.name] = round(n.allocation, 2)
    return results

def run_rebalance(securities_df: pd.DataFrame) -> pd.DataFrame:
    roots = build_hierarchy(securities_df)
    improved_waterfall_rebalance(roots)
    allocations = collect_leaf_allocations(roots)
    result = pd.DataFrame(list(allocations.items()), columns=['Ticker', 'Allocation'])
    return result

# --- Streamlit Web App ---
st.title("Constrained Portfolio Rebalancing")
st.write("Upload an Excel or CSV file with columns: Ticker, risk, asset_class, target, constrained.")

# Provide a downloadable template
template_df = pd.DataFrame(columns=["Ticker", "risk", "asset_class", "target", "constrained"])
template_buffer = BytesIO()
with pd.ExcelWriter(template_buffer, engine="openpyxl") as writer:
    template_df.to_excel(writer, index=False, sheet_name="Template")
template_buffer.seek(0)
template_bytes = template_buffer.getvalue()
st.download_button(
    "Download Excel template",
    data=template_bytes,
    file_name="rebalance_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        result = run_rebalance(df)
        # Merge original metadata and rename Allocation to computed
        output_df = df[["Ticker", "risk", "asset_class", "target", "constrained"]].merge(
            result, on="Ticker"
        )
        output_df = output_df.rename(columns={"Allocation": "computed"})
        st.dataframe(output_df)
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download results as CSV",
            data=csv,
            file_name="rebalance_results.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- pytest Test ---
def test_minimal_sample():
    sample_data = [
        {'Ticker':'DGRO','risk':'Growth','asset_class':'US Large Cap Equity','target':0,'constrained':22000},
        {'Ticker':'IVV','risk':'Growth','asset_class':'US Large Cap Equity','target':20000,'constrained':None},
        {'Ticker':'BND','risk':'Defensive','asset_class':'US Bonds','target':10000,'constrained':None},
    ]
    sample_df = pd.DataFrame(sample_data)
    expected = {'DGRO':22000, 'IVV':0, 'BND':8000}
    result = run_rebalance(sample_df)
    comp = dict(zip(result['Ticker'], result['Allocation']))
    for ticker, exp in expected.items():
        assert comp[ticker] == exp
