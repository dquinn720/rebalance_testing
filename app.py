import pandas as pd
from typing import List
import streamlit as st

# --- Node Class and Rebalance Logic ---
class Node:
    def __init__(self, name: str, target: float, constraint: float, children: List['Node'] = None):
        self.name = name
        self.target = float(target)
        self.constraint = float(constraint) if constraint is not None else 0.0
        self.children = children or []
        self.allocation = 0.0

def waterfall_rebalance(nodes: List[Node]) -> None:
    total_target = sum(n.target for n in nodes)
    total_constraint = sum(n.constraint for n in nodes)
    extra = total_target - total_constraint
    overshoots = [max(n.constraint - n.target, 0.0) for n in nodes]
    overshoot_total = sum(overshoots)

    if overshoot_total > 0 and extra > 0:
        free_nodes = [n for n in nodes if n.constraint < n.target]
        sum_targets_free = sum(n.target for n in free_nodes)
        for n in nodes:
            if n.constraint > n.target:
                n.allocation = n.constraint
            else:
                n.allocation = n.target - (overshoot_total * (n.target / sum_targets_free))
    elif extra > 0:
        head_nodes = [n for n in nodes if n.constraint < n.target]
        head_sum = sum(n.target - n.constraint for n in head_nodes)
        for n in nodes:
            if n.constraint >= n.target:
                n.allocation = n.constraint
            else:
                n.allocation = n.constraint + ((n.target - n.constraint) * (extra / head_sum))
    else:
        for n in nodes:
            n.allocation = n.constraint

    for parent in nodes:
        if parent.children:
            child_total = sum(c.target for c in parent.children) or 1.0
            for c in parent.children:
                c.target = (c.target / child_total) * parent.allocation
            waterfall_rebalance(parent.children)

# --- Streamlit Web App ---
st.title("Constrained Portfolio Rebalancer")
st.write("Upload your Excel file to see the rebalanced results.")

# Optional download template
with open("Portfolio_Template.xlsx", "rb") as f:
    st.download_button("Download Excel Template", f, file_name="Portfolio_Template.xlsx")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")


    if not set(["Ticker", "Risk", "Asset Class", "Target", "Constraint"]).issubset(df.columns):
        st.error("Missing required columns in the Excel file.")
    else:
        securities = df.set_index("Ticker").T.to_dict()
        risk_map = {}
        for t, m in securities.items():
            risk_map.setdefault(m['Risk'], {}).setdefault(m['Asset Class'], []).append(t)

        roots = []
        for risk, acs in risk_map.items():
            ac_nodes = [Node(ac,
                             sum(securities[t]['Target'] for t in tickers),
                             sum(securities[t]['Constraint'] if pd.notnull(securities[t]['Constraint']) else 0 for t in tickers),
                             [Node(t, securities[t]['Target'], securities[t]['Constraint']) for t in tickers])
                        for ac, tickers in acs.items()]
            roots.append(Node(risk,
                              sum(a.target for a in ac_nodes),
                              sum(a.constraint for a in ac_nodes),
                              ac_nodes))

        waterfall_rebalance(roots)

        rows = []
        for r in roots:
            for ac in r.children:
                for s in ac.children:
                    rows.append({
                        'Ticker': s.name,
                        'Computed Allocation': round(s.allocation, 2),
                        'Target': s.target,
                        'Constraint': s.constraint
                    })
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df)
        st.download_button("Download Result as CSV", result_df.to_csv(index=False), "Rebalanced_Portfolio.csv")
