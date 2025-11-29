from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PolicyScale:
    id: str
    name: str
    column: str
    levels: Dict[int, str]
    has_flag: bool

POLICY_SCALES: Dict[str, PolicyScale] = {
    # --- C: Containment and closure policies ---
    "C1": PolicyScale(
        id="C1",
        name="School closing",
        column="C1M_School closing",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommend closing or significant operational changes",
            2: "Require closing of some levels / categories",
            3: "Require closing of all levels",
        },
    ),
    "C2": PolicyScale(
        id="C2",
        name="Workplace closing",
        column="C2M_Workplace closing",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommend closing / work from home or major changes",
            2: "Require closing for some sectors / categories",
            3: "Require closing for all-but-essential workplaces",
        },
    ),
    "C3": PolicyScale(
        id="C3",
        name="Cancel public events",
        column="C3M_Cancel public events",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommend cancelling public events",
            2: "Require cancelling public events",
        },
    ),
    "C4": PolicyScale(
        id="C4",
        name="Restrictions on gatherings",
        column="C4M_Restrictions on gatherings",
        has_flag=True,
        levels={
            0: "No restrictions",
            1: "Limits only very large gatherings (>1000 people)",
            2: "Limits gatherings 101–1000 people",
            3: "Limits gatherings 11–100 people",
            4: "Limits gatherings of 10 people or less",
        },
    ),
    "C5": PolicyScale(
        id="C5",
        name="Close public transport",
        column="C5M_Close public transport",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommend closing / heavily reduce service",
            2: "Require closing / prohibit most citizens",
        },
    ),
    "C6": PolicyScale(
        id="C6",
        name="Stay at home requirements",
        column="C6M_Stay at home requirements",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommend not leaving house",
            2: "Require not leaving house with broad exceptions",
            3: "Require not leaving house with minimal exceptions",
        },
    ),
    "C7": PolicyScale(
        id="C7",
        name="Restrictions on internal movement",
        column="C7M_Restrictions on internal movement",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommend not to travel between regions/cities",
            2: "Internal movement restrictions in place",
        },
    ),
    "C8": PolicyScale(
        id="C8",
        name="Restrictions on international travel",
        # Note: there is no "M" variant; the uninterrupted series is C8EV
        column="C8EV_International travel controls",
        has_flag=False,
        levels={
            0: "No restrictions",
            1: "Screening arrivals",
            2: "Quarantine arrivals from some or all regions",
            3: "Ban arrivals from some regions",
            4: "Ban on all regions or total border closure",
        },
    ),

    # --- E: Economic policies (ordinal only) ---
    "E1": PolicyScale(
        id="E1",
        name="Income support",
        column="E1_Income support",
        has_flag=True,  # flag distinguishes formal vs all workers
        levels={
            0: "No income support",
            1: "Government replaces <50% of lost salary",
            2: "Government replaces ≥50% of lost salary",
        },
    ),
    "E2": PolicyScale(
        id="E2",
        name="Debt / contract relief",
        column="E2_Debt/contract relief",
        has_flag=False,
        levels={
            0: "No debt/contract relief",
            1: "Narrow relief (single contract type)",
            2: "Broad debt/contract relief",
        },
    ),

    # --- H: Health system policies ---
    "H1": PolicyScale(
        id="H1",
        name="Public information campaign",
        column="H1_Public information campaigns",
        has_flag=True,
        levels={
            0: "No campaign",
            1: "Officials urge caution",
            2: "Coordinated public info campaign",
        },
    ),
    "H2": PolicyScale(
        id="H2",
        name="Testing policy",
        column="H2_Testing policy",
        has_flag=False,
        levels={
            0: "No testing policy",
            1: "Testing only for symptomatic + specific criteria",
            2: "Testing for anyone with symptoms",
            3: "Open public testing (including asymptomatic)",
        },
    ),
    "H3": PolicyScale(
        id="H3",
        name="Contact tracing",
        column="H3_Contact tracing",
        has_flag=False,
        levels={
            0: "No contact tracing",
            1: "Limited contact tracing (not all cases)",
            2: "Comprehensive contact tracing (all cases)",
        },
    ),
    "H6": PolicyScale(
        id="H6",
        name="Facial coverings",
        column="H6M_Facial coverings",
        has_flag=True,
        levels={
            0: "No policy",
            1: "Recommended",
            2: "Required in some shared/public spaces",
            3: "Required in all shared/public spaces with others",
            4: "Required outside home at all times",
        },
    ),
    "H7": PolicyScale(
        id="H7",
        name="Vaccination policy",
        column="H7_Vaccination policy",
        has_flag=True,  # flag = cost (0 at-cost, 1 low/no cost)
        levels={
            0: "No availability",
            1: "Availability for ONE of: key workers / clinically vulnerable (non-elderly) / elderly",
            2: "Availability for TWO of those groups",
            3: "Availability for ALL THREE groups",
            4: "All three plus additional broad groups",
            5: "Universal availability",
        },
    ),
    "H8": PolicyScale(
        id="H8",
        name="Protection of elderly people",
        column="H8M_Protection of elderly people",
        has_flag=True,
        levels={
            0: "No measures",
            1: "Recommended isolation / hygiene / visitor restrictions",
            2: "Narrow restrictions (LTCFs or some home protections)",
            3: "Extensive restrictions (LTCFs and/or home)",
        },
    ),
}

# mapping from dataframe column name → indicator ID
COLUMN_TO_POLICY_ID: Dict[str, str] = {
    scale.column: pid for pid, scale in POLICY_SCALES.items()
}

# strictest level per policy ID
STRICTEST_LEVEL: Dict[str, int] = {
    pid: max(scale.levels.keys()) for pid, scale in POLICY_SCALES.items()
}

LAXEST_LEVEL : Dict[str, int] = {
    pid : min(scale.levels.keys()) for pid, scale in POLICY_SCALES.items()
}

def get_strictest_action_for_columns(columns: list[str]) -> dict[str, int]:
    action: dict[str, int] = {}
    for col in columns:
        pid = COLUMN_TO_POLICY_ID.get(col)
        if pid is None:
            continue
        max_level = STRICTEST_LEVEL[pid]
        action[col] = max_level
    return action

def get_laxest_action_for_columns(columns: list[str]) -> dict[str, int]:
    action: dict[str, int] = {}
    for col in columns:
        pid = COLUMN_TO_POLICY_ID.get(col)
        if pid is None:
            continue
        lax_level = LAXEST_LEVEL[pid]
        action[col] = lax_level
    return action
