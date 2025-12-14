from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from extractors import (
    parse_eml,
    detect_ext_kind,
    extract_rows_from_body_text,
    extract_rows_from_attachment,
    normalize_row_keys,
    compute_confidence,
    validate_and_clean_rows,
)
from llm_providers import Providers
from negotiate import build_targets, draft_email_for_carrier


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip())
    return (s[:80] or "UNKNOWN")


def extract_carrier_name(carrier_or_subject: str) -> str:
    """
    Extract clean carrier name from either:
    - Direct carrier name: "Carrier Alpha"
    - Email subject: "Rate Offer â€“ Carrier Kappa (Intra-Asia)"
    Returns just the carrier name portion.
    """
    s = (carrier_or_subject or "").strip()

    # Pattern 1: "Rate Offer â€“ Carrier XYZ (...)" -> "Carrier XYZ"
    # Pattern 2: "Carrier XYZ" -> "Carrier XYZ"
    match = re.search(r'(Carrier\s+\w+)', s, re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback: return as-is
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input directory containing .eml files")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--conf_threshold", type=float, default=float(os.getenv("CONF_THRESHOLD", "0.55")))
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug or os.getenv("DEBUG", "0") == "1" else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist logs to output directory for debugging
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"Starting processing: input={in_dir}, output={out_dir}")

    providers = Providers.from_env()
    logger.info(f"Providers initialized: openai_enabled={providers.openai_enabled}, "
                f"vision={providers.vision_provider_name}, text={providers.text_provider_name}")

    emls = sorted(in_dir.glob("*.eml"))
    logger.info(f"Found {len(emls)} .eml files to process")

    run_report: Dict[str, Any] = {
        "started_at": now_iso(),
        "openai_enabled": providers.openai_enabled,
        "vision_provider": providers.vision_provider_name,
        "text_provider": providers.text_provider_name,
        "llm_usage": {
            "openai": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0},
            "hf": {"calls": 0},
        },
        "files": [],
        "counts": {"quotes_rows": 0, "valid_rows": 0, "review_items": 0},
    }

    normalized_rows: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []

    for eml_path in emls:
        logger.info(f"Processing: {eml_path.name}")
        carrier_guess, body_text, attachments = parse_eml(eml_path)
        logger.debug(f"  Carrier: {carrier_guess}, Body length: {len(body_text)}, Attachments: {len(attachments)}")

        file_log = {
            "file": eml_path.name,
            "carrier_guess": carrier_guess,
            "attachments": len(attachments),
            "events": [],
        }

        # 1) Email body
        if body_text.strip():
            logger.debug(f"  Extracting from email body...")
            try:
                rows, method, warn, usage = extract_rows_from_body_text(
                    body_text,
                    providers=providers,
                    conf_threshold=args.conf_threshold,
                )
                providers.accumulate_usage(run_report["llm_usage"], usage)

                rows = [normalize_row_keys(r) for r in rows]
                conf, cw = compute_confidence(rows)

                # Strict validation/cleaning
                valid_rows, val_warn, val_rejects = validate_and_clean_rows(rows)
                logger.info(f"  Email body: {len(rows)} rows extracted, {len(valid_rows)} valid (method={method}, conf={conf:.2f})")

                file_log["events"].append(
                    {
                        "stage": "email_body",
                        "rows": len(rows),
                        "valid_rows": len(valid_rows),
                        "method": method,
                        "conf": conf,
                        "warnings": warn + cw + val_warn,
                    }
                )

                # Save rejected rows for review
                for rej in val_rejects:
                    review_items.append(
                        {
                            "source": str(eml_path),
                            "reason": "row_validation_failed",
                            "detail": "email_body",
                            "snippet": rej.get("why", "")[:800],
                            "partial": rej.get("row", {}),
                        }
                    )

                for r in valid_rows:
                    normalized_rows.append(
                        {
                            "carrier": extract_carrier_name(r.get("carrier") or carrier_guess),
                            "source": f"{eml_path}",
                            "extraction_method": method,
                            "confidence": conf,
                            **r,
                        }
                    )

            except Exception as e:
                logger.error(f"  Email body extraction failed: {e}")
                review_items.append(
                    {
                        "source": str(eml_path),
                        "reason": "email_body_parse_failed",
                        "detail": str(e),
                        "snippet": body_text[:800],
                    }
                )

        # 2) Attachments (always)
        for (filename, data) in attachments:
            kind = detect_ext_kind(filename)
            logger.debug(f"  Processing attachment: {filename} (type={kind})")
            try:
                rows, method, warn, usage = extract_rows_from_attachment(
                    filename,
                    data,
                    providers=providers,
                    conf_threshold=args.conf_threshold,
                )
                providers.accumulate_usage(run_report["llm_usage"], usage)

                rows = [normalize_row_keys(r) for r in rows]
                conf, cw = compute_confidence(rows)

                valid_rows, val_warn, val_rejects = validate_and_clean_rows(rows)
                logger.info(f"  Attachment {filename}: {len(rows)} rows extracted, {len(valid_rows)} valid (method={method}, conf={conf:.2f})")

                file_log["events"].append(
                    {
                        "stage": "attachment",
                        "name": filename,
                        "kind": kind,
                        "rows": len(rows),
                        "valid_rows": len(valid_rows),
                        "method": method,
                        "conf": conf,
                        "warnings": warn + cw + val_warn,
                    }
                )

                if not rows:
                    logger.warning(f"  Attachment {filename} yielded no rows")
                    review_items.append(
                        {
                            "source": str(eml_path),
                            "reason": "attachment_no_rows",
                            "detail": f"{filename} ({kind})",
                            "snippet": "",
                        }
                    )
                    continue

                for rej in val_rejects:
                    review_items.append(
                        {
                            "source": str(eml_path),
                            "reason": "row_validation_failed",
                            "detail": f"attachment::{filename}",
                            "snippet": rej.get("why", "")[:800],
                            "partial": rej.get("row", {}),
                        }
                    )

                for r in valid_rows:
                    normalized_rows.append(
                        {
                            "carrier": extract_carrier_name(r.get("carrier") or carrier_guess),
                            "source": f"{eml_path}::{filename}",
                            "extraction_method": method,
                            "confidence": conf,
                            **r,
                        }
                    )
            except Exception as e:
                logger.error(f"  Attachment {filename} extraction failed: {e}")
                review_items.append(
                    {
                        "source": str(eml_path),
                        "reason": "attachment_parse_failed",
                        "detail": f"{filename}: {e}",
                        "snippet": "",
                    }
                )

        run_report["files"].append(file_log)

    logger.info(f"Saving results: {len(normalized_rows)} normalized rows")

    def clean_for_parquet(rows):
        """Ensure numeric fields are actually numeric for Parquet compatibility."""
        import re
        
        def tonum(val):
            if val is None or val == "":
                return None
            if isinstance(val, (int, float)):
                return float(val)
            s = str(val).strip()
            # Remove common non-numeric parts
            s = s.replace(",", "").replace(" ", "").replace("USD", "")
            s = s.replace("days", "").replace("day", "").strip()
            try:
                return float(s)
            except:
                # Extract first number
                match = re.search(r'-?\d+\.?\d*', s)
                return float(match.group(0)) if match else None
        
        for row in rows:
            # Force these fields to be numeric
            row["transittimedays"] = tonum(row.get("transittimedays"))
            row["ofrateusd"] = tonum(row.get("ofrateusd"))
            row["freedaysorigin"] = tonum(row.get("freedaysorigin"))
            row["freedaysdestination"] = tonum(row.get("freedaysdestination"))
            row["containercount"] = tonum(row.get("containercount"))
            
            # FIX: Convert any remaining numeric-looking columns
            for key in list(row.keys()):
                if key in ["FD-O", "FD-D", "FDO", "FDD"]:
                    row[key] = tonum(row[key])
        
        return rows


    normalized_rows = clean_for_parquet(normalized_rows)
    df = pd.DataFrame(normalized_rows)
    (out_dir / "normalized_quotes.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    try:
        # Convert all object columns - coerce numeric ones, stringify the rest
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try numeric conversion first (for columns that should be numbers)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # If still object type (has non-numeric data), convert to clean strings
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).replace(['None', 'nan', '', 'NaN'], None)
        
        df.to_parquet(out_dir / "normalized_quotes.parquet", index=False)
        logger.info("Parquet file saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save parquet file: {e}")


    # Draft negotiation emails per carrier
    drafts_dir = out_dir / "drafts"
    drafts_dir.mkdir(exist_ok=True)

    by_carrier: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        by_carrier[str(row.get("carrier") or "UNKNOWN")].append(row)

    logger.info(f"Building targets and drafting emails for {len(by_carrier)} carriers")
    targets = build_targets(normalized_rows)

    for carrier, rows in by_carrier.items():
        logger.debug(f"  Drafting email for carrier: {carrier} ({len(rows)} rows)")
        draft = draft_email_for_carrier(carrier, rows, targets, providers=providers)
        (drafts_dir / f"{safe_name(carrier)}.md").write_text(draft, encoding="utf-8")

    (out_dir / "review_items.json").write_text(json.dumps(review_items, indent=2), encoding="utf-8")

    run_report["finished_at"] = now_iso()
    run_report["counts"]["quotes_rows"] = len(df)
    run_report["counts"]["valid_rows"] = len(normalized_rows)
    run_report["counts"]["review_items"] = len(review_items)
    (out_dir / "run_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    logger.info(f"Processing complete!")
    logger.info(f"  Valid rows: {len(normalized_rows)}")
    logger.info(f"  Review items: {len(review_items)}")
    logger.info(f"  LLM usage: {run_report['llm_usage']}")
    print(f"Done. Valid rows={len(normalized_rows)} Review={len(review_items)} Output={out_dir}")


if __name__ == "__main__":
    main()