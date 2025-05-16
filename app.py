import streamlit as st
import pandas as pd
from typing import List

# --- Algorithm Implementation ---
class Node:
    def __init__(self, name: str, target: float, constraint: float, children: List['Node'] = None):
        self.name       = name
        self.target     = float(target)
        self.constraint = float(constraint) if constraint is not None else 0.0
        self.children   = children or []
        self.allocation = 0.0


def waterfall_rebalance(nodes: List[Node]) -> None:
    total_target     = sum(n.target     for n in nodes)
    total_constraint = sum(n.constraint for n in nodes)
    extra = total_target - total_constraint
    # 1) Distribute headroom proportionally if any
    if extra > 0:
        head_sum = sum((n.target - n.constraint) for n in nodes if n.target > n.constraint)
        for n in nodes:
            if n.target <= n.constraint:
                n.allocation = n.constraint
            else:
                n.allocation = n.constraint + ((n.target - n.constraint) * (extra / head_sum))
    # 2) No headroom: everyone at constraint
    else:
        for n in nodes:
            n.allocation = n.constraint
    # Recurse into children
    for parent in nodes:
        if parent.children:
            child_total = sum(c.target for c in parent.children) or 1.0
            for c in parent.children:
                c.target = (c.target / child_total) * parent.allocation
            waterfall_rebalance(parent.children)


def run_rebalance(securities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the constrained rebalance algorithm on an input DataFrame.
    Expects columns: Ticker, risk, asset_class, target, constrained
    Returns a DataFrame with columns: Ticker, Allocation
    """
    # Build a dict of security metadata
    sec = {
        row['Ticker']: {
            'risk': row['risk'],
            'asset_class': row['asset_class'],
            'target': row['target'],
            'constrained': row['constrained']
        }
        for _, row in securities_df.iterrows()
    }
    # Group by risk and asset class
    risk_map = {}
    for ticker, meta in sec.items():
        risk_map.setdefault(meta['risk'], {})
        risk_map[meta['risk']].setdefault(meta['asset_class'], []).append(ticker)
    # Build the hierarchy of Nodes
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
    # Execute the waterfall rebalance
    waterfall_rebalance(roots)
    # Extract allocations into a DataFrame
    rows = []
    for risk_node in roots:
        for ac_node in risk_node.children:
            for sec_node in ac_node.children:
                rows.append({
                    'Ticker': sec_node.name,
                    'Allocation': round(sec_node.allocation, 2)
                })
    return pd.DataFrame(rows)

# --- Streamlit Web App ---
st.title("Constrained Portfolio Rebalancing")
st.write("Upload an Excel file with columns: Ticker, risk, asset_class, target, constrained.")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded:
    try:
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
# To run: pytest app.py

def test_minimal_sample():
    # Minimal sample from notebook
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
