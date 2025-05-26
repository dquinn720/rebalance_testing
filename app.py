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


def build_hierarchy(data: dict):
    by_risk = {}
    for sec, info in data.items():
        risk = info['risk']
        asset_class = info['asset_class']
        sec_node = Node(sec, info['target'], info.get('constrained', None), info.get('holding', None))
        if risk not in by_risk:
            by_risk[risk] = {}
        if asset_class not in by_risk[risk]:
            by_risk[risk][asset_class] = []
        by_risk[risk][asset_class].append(sec_node)
    risk_nodes = []
    for risk, ac_dict in by_risk.items():
        ac_nodes = []
        for ac, sec_nodes in ac_dict.items():
            target = sum(s.target for s in sec_nodes)
            constraint = sum(s.constraint for s in sec_nodes)
            holding = sum(s.holding for s in sec_nodes)
            ac_nodes.append(Node(ac, target, constraint, holding, children=sec_nodes))
        target = sum(a.target for a in ac_nodes)
        constraint = sum(a.constraint for a in ac_nodes)
        holding = sum(a.holding for a in ac_nodes)
        risk_nodes.append(Node(risk, target, constraint, holding, children=ac_nodes))
    return risk_nodes


## 2. Proportional Cascading Overshoot Waterfall Algorithm

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

def waterfall_with_min_constraint(nodes: List[Node]) -> None:
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
            waterfall_with_min_constraint(parent.children)


def waterfall_with_max_constraint(
    nodes: List[Node],
    parent_alloc: Optional[float] = None,
    tol: float = 1e-6
) -> None:
    """
    1) Scale each node’s target to sum to `parent_alloc`
    2) Cap at each node’s holding
    3) Redistribute any “over-caps” among those with spare capacity
    4) Recurse into children with each node’s final allocation
    """
    # 1) decide how much to split
    level_target = parent_alloc if parent_alloc is not None else sum(n.target for n in nodes)
    sum_tgts      = sum(n.target for n in nodes) or 1.0

    # Scale
    for n in nodes:
        n.allocation = (n.target / sum_tgts) * level_target

    # Cap & collect removed
    removed = 0.0
    for n in nodes:
        if n.allocation > n.holding:
            removed        += (n.allocation - n.holding)
            n.allocation   = n.holding

    # Redistribute removed to any sibling with capacity
    fill      = removed
    eligible  = [n for n in nodes if n.allocation < n.holding]
    while fill > tol and eligible:
        caps    = [n.holding - n.allocation for n in eligible]
        cap_sum = sum(caps) or 1.0
        new_el  = []
        leftover = 0.0

        for n, cap in zip(eligible, caps):
            add = fill * (cap / cap_sum)
            if n.allocation + add > n.holding:
                leftover    += (n.allocation + add) - n.holding
                n.allocation = n.holding
            else:
                n.allocation += add
                new_el.append(n)

        fill     = leftover
        eligible = new_el

    # 4) recurse
    for n in nodes:
        if n.children:
            waterfall_with_max_constraint(n.children, parent_alloc=n.allocation, tol=tol)


def collect_leaf_allocations(nodes, results=None):
    if results is None:
        results = {}
    for n in nodes:
        if n.children:
            collect_leaf_allocations(n.children, results)
        else:
            results[n.name] = round(n.allocation, 2)
    return results


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


def full_rebalance(input_dict,cash=0):
    data = targets_pct_to_dollars(input_dict,cash)
    tree = build_hierarchy(data)
    waterfall_with_min_constraint(tree)
    allocations = collect_leaf_allocations(tree)
    output = []
    for ticker, meta in data.items():
        output.append({
            "Ticker": ticker,
            "Risk": meta["risk"],
            "Asset Class": meta["asset_class"],
            "Target": meta.get("target"),
            "Constrained": meta.get("constrained"),
            "Allocation": round(allocations.get(ticker, 0.0), 2),
            "Holding": meta.get("holding"),
            "Trade": round(allocations.get(ticker, 0.0) - (meta.get("holding") or 0.0), 2)
        })
    return sorted(output, key=lambda x: x["Ticker"])


def buy_only_rebalance(input_dict: dict, cash: float = 0) -> List[dict]:
    data = targets_pct_to_dollars(input_dict, cash)
    tree = build_hierarchy(data)
    waterfall_with_min_constraint(tree)
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
    waterfall_with_min_constraint(tree2)
    second = collect_leaf_allocations(tree2)
    out = []
    for t, m in second_input.items():
        out.append({
            'Ticker': t,
            'Risk': m['risk'],
            'Asset Class': m['asset_class'],
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
    waterfall_with_min_constraint(tree)
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
    waterfall_with_max_constraint(tree2)
    second = collect_leaf_allocations(tree2)
    out = []
    for t, m in second_input.items():
        out.append({
            'Ticker': t,
            'Risk': m['risk'],
            'Asset Class': m['asset_class'],
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
            out_df = out_df.sort_values(by=['Risk','Asset Class','Target'], ascending=[True, True, False]) 

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
            risk_df = out_df.groupby('Risk')[['Target','Holding','Allocation']].sum()
            st.bar_chart(risk_df)
            st.write("### Weight by Asset Class")
            ac_df = out_df.groupby('Asset Class')[['Target','Holding','Allocation']].sum()
            st.bar_chart(ac_df)
            
            st.write("### Weight by Ticker (grouped by Risk and Asset Class)")
            ticker_df = out_df.sort_values(['Risk','Asset Class','Ticker'])
            ticker_df = ticker_df.set_index('Ticker')[['Target','Holding','Allocation']]
            st.bar_chart(ticker_df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
