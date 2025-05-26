import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

# --- Node & Hierarchy Utilities ---
from typing import List, Optional

class Node:
    def __init__(self, name, target, constraint=None, holding=None, children: Optional[List['Node']] = None):
        self.name = name
        self.target = float(target)
        self.constraint = float(constraint) if constraint is not None else 0.0
        self.holding = float(holding) if holding is not None else 0.0
        self.allocation = 0.0
        self.children = children or []
    def __repr__(self):
        return f"Node({self.name}, {self.target}, {self.constraint}, {self.holding}, alloc={self.allocation})"


def build_hierarchy(df_dict: dict) -> List[Node]:
    # Build nodes grouped by risk and asset_class
    sec = df_dict
    risk_map = {}
    for ticker, meta in sec.items():
        risk_map.setdefault(meta['risk'], {}).setdefault(meta['asset_class'], []).append(ticker)
    roots = []
    for risk, acs in risk_map.items():
        ac_nodes = []
        for ac, tickers in acs.items():
            tgt = sum(sec[t]['target'] for t in tickers)
            con = sum(sec[t].get('constrained', 0) or 0 for t in tickers)
            children = [Node(t, sec[t]['target'], sec[t].get('constrained'), sec[t].get('holding')) for t in tickers]
            ac_nodes.append(Node(ac, tgt, con, None, children))
        total_tgt = sum(n.target for n in ac_nodes)
        total_con = sum(n.constraint for n in ac_nodes)
        roots.append(Node(risk, total_tgt, total_con, None, ac_nodes))
    return roots


def proportional_cascading_overshoot(nodes: List[Node], overshoot: float):
    eligible = [n for n in nodes if n.constraint < n.target]
    while overshoot > 1e-6 and eligible:
        sum_targets = sum(n.target for n in eligible)
        next_eligible = []
        redistributed = 0.0
        for n in eligible:
            reduction = overshoot * (n.target / sum_targets) if sum_targets else 0
            new_alloc = n.allocation - reduction
            if new_alloc < n.constraint:
                redistributed += (n.constraint - new_alloc)
                n.allocation = n.constraint
            else:
                n.allocation = new_alloc
                next_eligible.append(n)
        overshoot = redistributed
        eligible = next_eligible


def improved_waterfall_rebalance(nodes: List[Node]) -> None:
    total_target = sum(n.target for n in nodes)
    total_con = sum(n.constraint for n in nodes)
    extra = total_target - total_con
    # initial allocation
    for n in nodes:
        n.allocation = n.constraint if n.constraint >= n.target else n.target
    # cascade overshoot
    if extra > 0:
        overshoot = sum(max(n.constraint - n.target, 0) for n in nodes)
        if overshoot > 0:
            proportional_cascading_overshoot(nodes, overshoot)
    # recurse
    for p in nodes:
        if p.children:
            child_total = sum(c.target for c in p.children) or 1
            for c in p.children:
                c.target = c.target / child_total * p.allocation
            improved_waterfall_rebalance(p.children)


def collect_leaf_allocations(nodes: List[Node], res=None):
    if res is None:
        res = {}
    for n in nodes:
        if n.children:
            collect_leaf_allocations(n.children, res)
        else:
            res[n.name] = round(n.allocation, 2)
    return res


def targets_pct_to_dollars(data_dict: dict, cash: float = 0) -> dict:
    total_holding = sum(float(meta.get('holding') or 0) for meta in data_dict.values()) + cash
    new = {}
    for t, meta in data_dict.items():
        m = meta.copy()
        tgt = float(meta.get('target') or 0)
        if 0 <= tgt <= 1:
            m['target'] = round(tgt * total_holding, 2)
        else:
            m['target'] = tgt
        new[t] = m
    return new


def full_rebalance(input_dict: dict, cash: float = 0) -> List[dict]:
    data = targets_pct_to_dollars(input_dict, cash)
    tree = build_hierarchy(data)
    improved_waterfall_rebalance(tree)
    allocs = collect_leaf_allocations(tree)
    out = []
    for t, m in data.items():
        out.append({
            'Ticker': t,
            'risk': m['risk'],
            'asset_class': m['asset_class'],
            'Target': m['target'],
            'Constrained': m.get('constrained'),
            'Holding': m.get('holding'),
            'Allocation': allocs.get(t, 0),
            'Trade': round(allocs.get(t, 0) - (m.get('holding') or 0), 2)
        })
    return sorted(out, key=lambda x: x['Ticker'])


def buy_only_rebalance(input_dict: dict, cash: float = 0) -> List[dict]:
    data = targets_pct_to_dollars(input_dict, cash)
    tree = build_hierarchy(data)
    improved_waterfall_rebalance(tree)
    first = collect_leaf_allocations(tree)
    second_input = {}
    for t, m in data.items():
        second_input[t] = {
            'risk': m['risk'],
            'asset_class': m['asset_class'],
            'target': first.get(t, 0),
            'constrained': m.get('holding'),
            'holding': m.get('holding')
        }
    tree2 = build_hierarchy(second_input)
    improved_waterfall_rebalance(tree2)
    second = collect_leaf_allocations(tree2)
    out = []
    for t, m in second_input.items():
        out.append({
            'Ticker': t,
            'risk': m['risk'],
            'asset_class': m['asset_class'],
            'Target': m['target'],
            'Constrained': m.get('constrained'),
            'Holding': m.get('holding'),
            'Allocation': second.get(t, 0),
            'Trade': round(second.get(t, 0) - (m.get('holding') or 0), 2)
        })
    return sorted(out, key=lambda x: x['Ticker'])


def sell_only_rebalance(input_dict: dict, cash: float = 0) -> List[dict]:
    data = targets_pct_to_dollars(input_dict, cash)
    tree = build_hierarchy(data)
    improved_waterfall_rebalance(tree)
    first = collect_leaf_allocations(tree)
    second_input = {}
    for t, m in data.items():
        second_input[t] = {
            'risk': m['risk'],
            'asset_class': m['asset_class'],
            'target': first.get(t, 0),
            'constrained': m.get('constrained'),
            'holding': m.get('holding')
        }
    tree2 = build_hierarchy(second_input)
    improved_waterfall_rebalance(tree2)
    second = collect_leaf_allocations(tree2)
    out = []
    for t, m in second_input.items():
        out.append({
            'Ticker': t,
            'risk': m['risk'],
            'asset_class': m['asset_class'],
            'Target': m['target'],
            'Constrained': m.get('constrained'),
            'Holding': m.get('holding'),
            'Allocation': second.get(t, 0),
            'Trade': round(second.get(t, 0) - (m.get('holding') or 0), 2)
        })
    return sorted(out, key=lambda x: x['Ticker'])

# --- Streamlit App ---
st.title("Constrained Portfolio Rebalancing")
st.write("Upload a CSV or Excel file with columns: Ticker, risk, asset_class, target (in %), constrained (in dollars), holding (in dollars).")

# Downloadable template
cols = ["Ticker", "risk", "asset_class", "target (in %)", "constrained (in $)", "holding (in $)"]
template_df = pd.DataFrame(columns=cols)
template_buffer = BytesIO()
with pd.ExcelWriter(template_buffer, engine="openpyxl") as writer:
    template_df.to_excel(writer, index=False, sheet_name="Template")
template_buffer.seek(0)
st.download_button(
    "Download template",
    data=template_buffer.getvalue(),
    file_name="rebalance_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv","xlsx"] )

operation = st.selectbox("Select operation", ["Full Rebalance", "Raise Cash (Sell Only)", "Use Cash (Buy Only)"])
cash = st.number_input("Cash amount ($)", value=0.0, step=0.01)

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        input_dict = {
            row['Ticker']: {
                'risk': row['risk'],
                'asset_class': row['asset_class'],
                'target': row['target (in %)'],
                'constrained': row.get('constrained (in $)', 0),
                'holding': row.get('holding (in $)', 0)
            }
            for _, row in df.iterrows()
        }
        if st.button("Run"):
            if operation == "Full Rebalance":
                output = full_rebalance(input_dict, cash)
            elif operation == "Raise Cash (Sell Only)":
                output = sell_only_rebalance(input_dict, cash)
            else:
                output = buy_only_rebalance(input_dict, cash)
            out_df = pd.DataFrame(output)
            # Ensure grouping columns
            out_df = out_df.set_index('Ticker')
            meta = df[['Ticker','risk','asset_class']].set_index('Ticker')
            out_df = out_df.join(meta).reset_index()

            st.dataframe(out_df)
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name="rebalance_results.csv",
                mime="text/csv"
            )
            # Visuals
            st.write("### Weight by Risk")
            risk_df = out_df.groupby('risk')[['Target','Holding','Allocation']].sum()
            st.bar_chart(risk_df)
            st.write("### Weight by Asset Class")
            ac_df = out_df.groupby('asset_class')[['Target','Holding','Allocation']].sum()
            st.bar_chart(ac_df)
            
            st.write("### Weight by Ticker (grouped by Risk and Asset Class)")
            ticker_df = out_df.sort_values(['risk','asset_class','Ticker'])
            ticker_df = ticker_df.set_index('Ticker')[['Target','Holding','Allocation']]
            st.bar_chart(ticker_df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
