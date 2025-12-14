from __future__ import annotations

import statistics
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional, Set

from llm_providers import Providers


# -----------------------------
# Port Validation (UN/LOCODE)
# -----------------------------
def normalize_unlocode(code: str) -> str:
    """Normalize port code to UN/LOCODE-like token (strip non-alphanumerics)."""
    return re.sub(r"[^A-Z0-9]", "", (code or "").strip().upper())


def is_valid_unlocode(code: str) -> bool:
    """Validates UN/LOCODE format: 2 letters + 3 alphanumeric (e.g., CNSHA, USNYC)."""
    c = normalize_unlocode(code)
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{3}$", c)) if len(c) == 5 else False


def validate_ports_in_rows(rows: List[Dict[str, Any]]) -> List[str]:
    """Returns a list of warnings for invalid port codes."""
    warnings: List[str] = []
    seen_invalid: Set[Tuple[str, str]] = set()

    for r in rows:
        lane = r.get("lane_id", "Unknown")
        for field in ["quoted_POL", "quoted_POD", "requested_POL", "requested_POD"]:
            raw = str(r.get(field) or "").strip()
            if not raw:
                continue
            norm = normalize_unlocode(raw)
            if norm and not is_valid_unlocode(norm):
                key = (field, norm)
                if key not in seen_invalid:
                    warnings.append(
                        f"Invalid UN/LOCODE: {field}='{raw}' (norm='{norm}') (Lane: {lane})"
                    )
                    seen_invalid.add(key)
    return warnings


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_num(val: Any) -> Optional[float]:
    """
    Safely converts a value to float.
    Handles strings like '22days', 'USD 500', etc.
    Returns None if conversion fails.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s_val = str(val).strip()
    try:
        return float(s_val)
    except ValueError:
        pass

    # Extract first number found (e.g., "22days" -> 22, "$450" -> 450)
    match = re.search(r"-?\d+(\.\d+)?", s_val)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


# -----------------------------
# Core Logic: Target Building
# -----------------------------
def build_targets(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Encodes the assignment market guidance explicitly:
    - Quotes in this round are around spot ±5% -> spot_est ≈ median quoted per lane (+container_type)
    - Cheapest quoted this round is ~30% higher than previous tender's cheapest
      => prev_cheapest_est ≈ current_cheapest / 1.30
    - Preferred free days >= 11 at origin and destination

    Targets are keyed by (carrier, lane_id, container_type) to avoid overwrites.
    """
    by_lane_ct_rates: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    # 1) Collect valid rates per (lane, container_type)
    for r in rows:
        lane = str(r.get("lane_id") or "").strip()
        ctype = str(r.get("container_type") or "").strip().upper()
        rate = clean_num(r.get("of_rate_usd"))
        if lane and ctype and rate is not None and rate > 0:
            by_lane_ct_rates[(lane, ctype)].append(rate)

    # 2) Benchmarks per (lane, container_type)
    spot_est: Dict[Tuple[str, str], float] = {}
    cheapest: Dict[Tuple[str, str], float] = {}
    prev_cheapest_est: Dict[Tuple[str, str], float] = {}

    for k, rates in by_lane_ct_rates.items():
        if not rates:
            continue
        spot_est[k] = statistics.median(rates)
        cheapest[k] = min(rates)
        prev_cheapest_est[k] = cheapest[k] / 1.30

    targets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    # 3) Build targets per (carrier, lane, container_type)
    for r in rows:
        carrier = str(r.get("carrier") or "UNKNOWN").strip()
        lane = str(r.get("lane_id") or "").strip()
        ctype = str(r.get("container_type") or "").strip().upper()
        rate = clean_num(r.get("of_rate_usd"))

        if not carrier or not lane or not ctype or rate is None or rate <= 0:
            continue

        key_lane = (lane, ctype)
        spot = float(spot_est.get(key_lane, rate))
        peer = float(cheapest.get(key_lane, rate))
        prev = float(prev_cheapest_est.get(key_lane, peer / 1.30))

        # --- Target logic (fixed) ---
        # Start from a discount vs quote and vs spot (long-term should be below spot)
        base = min(rate * 0.95, spot * 0.92)

        # Prior anchor: allow modest uplift vs prior cheapest (since market up), but don't force above peer/quote
        prev_floor = prev * 1.08  # tune 1.05–1.12

        # Competitive cap vs peers: do not exceed cheapest peer (or at most +1%)
        peer_cap = peer * 1.01

        # Combine: ensure at least prev_floor, but never above peer_cap nor above their quote
        target = max(base, prev_floor)
        target = min(target, peer_cap, rate)

        # Free days asks
        fdo = clean_num(r.get("free_days_origin"))
        fdd = clean_num(r.get("free_days_destination"))
        ask_fdo = (fdo is None) or (fdo < 11)
        ask_fdd = (fdd is None) or (fdd < 11)

        targets[(carrier, lane, ctype)] = {
            "target_of_rate_usd": round(target, 2),
            "spot_est_usd": round(spot, 2),
            "peer_cheapest_usd": round(peer, 2),
            "prev_cheapest_est_usd": round(prev, 2),
            "spot_band_low_usd": round(spot * 0.95, 2),
            "spot_band_high_usd": round(spot * 1.05, 2),
            "ask_free_days_origin_11": bool(ask_fdo),
            "ask_free_days_destination_11": bool(ask_fdd),
        }

    return targets


# -----------------------------
# Email Drafting Entrypoint
# -----------------------------
def draft_email_for_carrier(
    carrier: str,
    rows: List[Dict[str, Any]],
    targets: Dict[Tuple[str, str, str], Dict[str, Any]],
    providers: Optional[Providers] = None,
) -> str:
    """
    Main entry point for drafting the email.
    Validates ports before generating.
    """
    _ = validate_ports_in_rows(rows)  # no side effects; caller can log/record warnings if desired

    # Initialize providers if not passed
    if providers is None:
        try:
            providers = Providers.from_env()
        except Exception:
            providers = None

    # Use LLM if enabled
    if providers and getattr(providers, "openai_enabled", False):
        try:
            return _draft_email_llm(carrier, rows, targets, providers)
        except Exception as e:
            print(f"LLM generation failed for {carrier}, falling back to template. Error: {e}")
            return _draft_email_template(carrier, rows, targets)

    return _draft_email_template(carrier, rows, targets)


def _draft_email_llm(
    carrier: str,
    rows: List[Dict[str, Any]],
    targets: Dict[Tuple[str, str, str], Dict[str, Any]],
    providers: Providers,
) -> str:
    """
    Generates a negotiation email using the configured LLM.
    Professional, manager-level tone with strategic negotiation approach.
    """
    lane_summaries: List[Dict[str, Any]] = []
    for r in rows:
        lane = str(r.get("lane_id") or "").strip()
        if not lane:
            continue

        ctype = str(r.get("container_type") or "").strip().upper()
        if not ctype:
            continue

        t = targets.get((carrier, lane, ctype))
        if not t:
            continue

        pol = r.get("quoted_POL") or r.get("requested_POL") or ""
        pod = r.get("quoted_POD") or r.get("requested_POD") or ""

        asks: List[str] = []
        if t["ask_free_days_origin_11"] or t["ask_free_days_destination_11"]:
            asks.append("Free days: Request ≥11 days at origin and destination")
        if r.get("accept_operational_requirements") is False:
            asks.append("Operational requirements acceptance")
        if r.get("accept_payment_methods") is False:
            asks.append("Payment terms alignment")

        lane_summaries.append(
            {
                "Lane": lane,
                "Container_Type": ctype,
                "Container_Count": r.get("container_count"),
                "Route": f"{pol} → {pod}",
                "Your_Quote_USD": clean_num(r.get("of_rate_usd")),
                "Our_Target_USD": t["target_of_rate_usd"],
                "Market_Intelligence": {
                    "Current_Spot_Est": t["spot_est_usd"],
                    "Market_Range": f"{t['spot_band_low_usd']}–{t['spot_band_high_usd']}",
                    "Best_Competitive_Quote": t["peer_cheapest_usd"],
                    "Prior_Period_Baseline": t["prev_cheapest_est_usd"],
                },
                "Service_Terms": {
                    "Free_Days_Origin": r.get("free_days_origin"),
                    "Free_Days_Dest": r.get("free_days_destination"),
                    "Transit_Time": r.get("transit_time_days"),
                },
                "Required_Adjustments": asks,
            }
        )

    prompt = f"""You are a Senior Procurement Manager at Cargoo, responsible for ocean freight tender negotiations.

**Your Task:** Draft a professional, strategic negotiation email to {carrier}.

**Negotiation Context:**
- Tender Objective: Secure competitive, stable long-term ocean freight capacity
- Market Dynamics: Spot rates are elevated (~±5% volatility). Our analysis shows current quotes are ~30% above prior-period contracted rates.
- Strategic Goal: Bridge this gap through data-driven negotiation while maintaining strong carrier relationships.

**Tone & Style:**
- Professional and respectful (you value the partnership)
- Data-driven and transparent (share market intelligence to build trust)
- Firm but collaborative (clear expectations with room for mutual benefit)
- Concise and action-oriented (busy stakeholders appreciate brevity)

**Email Structure:**
1. **Opening:** Thank them for their submission.
2. **Market Context (2-3 sentences):** Briefly explain the market situation to frame your counteroffer.
3. **Lane-by-Lane Analysis:** For each lane:
   - **Lane & Route**
   - **Quote vs. Target**
   - **Rationale:** Use spot estimate, peer benchmarks, and prior period baseline
   - **Service Considerations:** free days / ops requirements / payment alignment
4. **Value Proposition:** Clear give/gets
5. **Next Steps:** Call-to-action with timeline
6. **Closing:** Sign off as "Cargoo Tender Team"

**Critical Requirements:**
- Use ONLY the data provided below. Do NOT invent numbers, routes, or carrier names.
- Do NOT recalculate any numeric values; use the provided numeric fields exactly as provided.
- Verify all port codes follow UN/LOCODE format (5 characters: 2 letters + 3 alphanumeric).
- Format numbers clearly (use thousand separators for readability, e.g., USD 3,200).
- Use Markdown: Headers (##), bullet points, **bold** for emphasis.

**Data for {carrier}:**
{json.dumps(lane_summaries, indent=2)}

**Output:** A complete, ready-to-send email in Markdown format."""

    client = providers._openai_client()
    model = providers.openai_text_model
    is_reasoning = providers.is_reasoning_model(model)

    kwargs: Dict[str, Any] = {}
    if not is_reasoning:
        kwargs["temperature"] = 0.2

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Empty LLM response")
    return content


def _draft_email_template(
    carrier: str,
    rows: List[Dict[str, Any]],
    targets: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> str:
    """Standard template fallback."""
    lines: List[str] = []
    lines.append(f"Subject: Tender Round – Counterproposal for {carrier}")
    lines.append("")
    lines.append(f"Hi {carrier} team,")
    lines.append("")
    lines.append("Thanks for sharing your offer. Based on market context and peer benchmarks in this tender round,")
    lines.append("we'd like to propose the following adjustments to move toward awardable long-term levels.")
    lines.append("")
    lines.append("Market reference used:")
    lines.append("- Quotes in this round are assumed to be around spot ±5%, so we estimate spot per lane from the median of received quotes.")
    lines.append("- The cheapest quote this round is assumed ~30% higher than the previous tender's cheapest, so prior cheapest is inferred as (current cheapest / 1.30).")
    lines.append("- Preferred free days: ≥11 at origin and destination.")
    lines.append("")

    for r in rows:
        lane = str(r.get("lane_id") or "").strip()
        if not lane:
            continue

        ctype = str(r.get("container_type") or "").strip().upper()
        if not ctype:
            continue

        t = targets.get((carrier, lane, ctype))
        if not t:
            continue

        pol = r.get("quoted_POL") or r.get("requested_POL") or ""
        pod = r.get("quoted_POD") or r.get("requested_POD") or ""
        rate = clean_num(r.get("of_rate_usd"))
        ccnt = r.get("container_count")

        lines.append(f"### Lane {lane} ({pol} → {pod})")
        lines.append(f"- Equipment/Volume: **{ctype} x {ccnt}**")
        lines.append(f"- Ops req accepted: {r.get('accept_operational_requirements')}")
        lines.append(f"- Payment accepted: {r.get('accept_payment_methods')}")

        if rate is not None:
            lines.append(f"- Your OF rate: **{rate:,.0f} USD**")
            lines.append(
                f"- Target counteroffer OF: **{t['target_of_rate_usd']:,.0f} USD** "
                f"(spot est.: {t['spot_est_usd']:,.0f} | spot band ±5%: {t['spot_band_low_usd']:,.0f}–{t['spot_band_high_usd']:,.0f} | "
                f"peer cheapest: {t['peer_cheapest_usd']:,.0f} | prior cheapest est.: {t['prev_cheapest_est_usd']:,.0f})"
            )

        fdo = clean_num(r.get("free_days_origin"))
        fdd = clean_num(r.get("free_days_destination"))
        lines.append(f"- Free days offered: origin={fdo} | destination={fdd}")

        asks: List[str] = []
        if t["ask_free_days_origin_11"] or t["ask_free_days_destination_11"]:
            asks.append("Increase free days to **≥11** at origin and destination")
        if r.get("accept_operational_requirements") is False:
            asks.append("Confirm acceptance of operational requirements")
        if r.get("accept_payment_methods") is False:
            asks.append("Confirm acceptance of payment terms")

        if asks:
            lines.append("- Give/gets: If you can meet the counteroffer level, we can prioritize allocation; please also:")
            lines.append("  - " + "\n  - ".join(asks))

        lines.append("")

    lines.append("If you can confirm the above, we can prioritize these lanes for award consideration.")
    lines.append("")
    lines.append("Best regards,")
    lines.append("Cargoo Tender Team")

    return "\n".join(lines)
