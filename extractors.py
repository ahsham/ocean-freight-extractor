from __future__ import annotations

import csv
import io
import os
import re
import statistics
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import pdfplumber
from PIL import Image

from llm_providers import Providers

# simplest OCR (optional)
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

# render PDF pages (needed for non-openai vision providers and for OCR on PDFs)
try:
    from pdf2image import convert_from_bytes
    PDF_RENDER_AVAILABLE = True
except Exception:
    convert_from_bytes = None
    PDF_RENDER_AVAILABLE = False

# docx
try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    Document = None
    DOCX_AVAILABLE = False


PORT_LIKE_RE = re.compile(r"^[A-Z]{5}$")


REQUIRED_FIELDS = [
    "lane_id",
    "POR",
    "requested_POL",
    "quoted_POL",
    "requested_POD",
    "quoted_POD",
    "FND",
    "container_type",
    "container_count",
    "accept_operational_requirements",
    "accept_payment_methods",
    "transit_time_days",
    "sailing_frequency",
    "of_rate_usd",
    "free_days_origin",
    "free_days_destination",
]


def parse_eml(path: Path) -> Tuple[str, str, List[Tuple[str, bytes]]]:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    subject = msg.get("subject", "") or ""
    carrier_guess = subject.strip() or "UNKNOWN"

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    body = part.get_content()
                except Exception:
                    body = (part.get_payload(decode=True) or b"").decode(errors="ignore")
                break
    else:
        try:
            body = msg.get_content()
        except Exception:
            body = (msg.get_payload(decode=True) or b"").decode(errors="ignore")

    atts: List[Tuple[str, bytes]] = []
    for part in msg.iter_attachments():
        fn = part.get_filename() or "attachment.bin"
        data = part.get_payload(decode=True) or b""
        atts.append((fn, data))
    return carrier_guess, body or "", atts


def detect_ext_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return "excel"
    if ext == ".csv":
        return "csv"
    if ext == ".pdf":
        return "pdf"
    if ext in (".png", ".jpg", ".jpeg"):
        return "image"
    if ext == ".docx":
        return "docx"
    return "other"

# Public entry points
def extract_rows_from_body_text(
    body_text: str,
    *,
    providers: Providers,
    conf_threshold: float,
) -> Tuple[List[Dict[str, Any]], str, List[str], Optional[Dict[str, Any]]]:
    warnings: List[str] = []
    usage: Optional[Dict[str, Any]] = None

    rows = _parse_markdown_table(body_text) or _parse_tsv_like(body_text) or _parse_csv_like(body_text)
    if rows:
        norm_rows = [normalize_row_keys(r) for r in rows]
        conf, cw = compute_confidence(norm_rows)
        if conf >= conf_threshold:
            return norm_rows, "body_deterministic", warnings + cw, usage

        # If deterministic parse produced rows but confidence is low, keep them if they validate
        valid_rows, val_warn, _ = validate_and_clean_rows(norm_rows)
        if valid_rows:
            return valid_rows, "body_deterministic", warnings + cw + ["body_low_confidence"] + val_warn, usage
        warnings += ["body_low_confidence"] + cw

    rows2, usage = providers.universal_parse_rows(body_text)
    rows2 = explode_columnar_objects(rows2)
    return rows2, "body_llm_universal", warnings, usage


def explode_columnar_objects(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If the model returns a single dict with column-wise lists OR semicolon-separated strings,
    convert to row-wise dicts.
    Example: {"Lane":[...], "POL":[...]} -> [{"Lane":..., "POL":...}, ...]
    Example: {"Lane":"L1; L2", "POL":"X; Y"} -> [{"Lane":"L1", "POL":"X"}, {"Lane":"L2", "POL":"Y"}]
    """
    if not rows or len(rows) != 1:
        return rows
    obj = rows[0]
    if not isinstance(obj, dict) or not obj:
        return rows

    # Check for list-based columns
    list_keys = [k for k, v in obj.items() if isinstance(v, list)]
    if list_keys:
        lengths = [len(obj[k]) for k in list_keys]
        if len(set(lengths)) != 1:
            return rows  # inconsistent columns

        n = lengths[0]
        out: List[Dict[str, Any]] = []
        for i in range(n):
            r: Dict[str, Any] = {}
            for k, v in obj.items():
                if isinstance(v, list):
                    r[k] = v[i] if i < len(v) else None
                else:
                    r[k] = v  # broadcast scalars
            out.append(r)
        return out

    # Check for semicolon-separated string columns
    semicolon_keys = [k for k, v in obj.items()
                      if isinstance(v, str) and '; ' in str(v)]
    if semicolon_keys:
        # Split all semicolon-separated values
        split_values = {}
        for k, v in obj.items():
            if isinstance(v, str) and '; ' in v:
                split_values[k] = [x.strip() for x in v.split('; ')]
            else:
                split_values[k] = [v]  # scalar becomes single-item list

        # Check if all lists have the same length
        lengths = [len(v) for v in split_values.values() if isinstance(v, list) and len(v) > 1]
        if not lengths or len(set(lengths)) != 1:
            return rows  # inconsistent lengths

        n = lengths[0]
        out: List[Dict[str, Any]] = []
        for i in range(n):
            r: Dict[str, Any] = {}
            for k, v_list in split_values.items():
                if len(v_list) > 1:
                    r[k] = v_list[i] if i < len(v_list) else None
                else:
                    r[k] = v_list[0]  # broadcast scalars
            out.append(r)
        return out

    return rows

def _parse_tsv_like(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if not text or "\n" not in text:
        return []
    lines = text.splitlines()
    start_idx = -1
    for i, ln in enumerate(lines):
        if ln.count("\t") >= 2:
            start_idx = i
            break
    if start_idx == -1:
        return []
    sliced = "\n".join(lines[start_idx:])
    try:
        reader = csv.DictReader(io.StringIO(sliced), delimiter="\t")
        rows = [dict(r) for r in reader]
        return [r for r in rows if any(str(v or "").strip() for v in r.values())]
    except Exception:
        return []

def extract_rows_from_attachment(
    filename: str,
    data: bytes,
    *,
    providers: Providers,
    conf_threshold: float,
) -> Tuple[List[Dict[str, Any]], str, List[str], Optional[Dict[str, Any]]]:
    warnings: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    kind = detect_ext_kind(filename)

    if kind == "excel":
        rows = _extract_excel(data)
        conf, cw = compute_confidence(rows)
        return rows, "excel", warnings + cw, usage

    if kind == "csv":
        rows = _extract_csv(data)
        conf, cw = compute_confidence(rows)
        return rows, "csv", warnings + cw, usage

    if kind == "docx":
        if not DOCX_AVAILABLE:
            raise RuntimeError("python-docx not installed (required for .docx)")
        text = _extract_docx_tables_to_text(data)
        # Try deterministic parse before LLM
        det_rows = _parse_markdown_table(text) or _parse_tsv_like(text) or _parse_csv_like(text)
        if det_rows:
            conf, cw = compute_confidence(det_rows)
            warnings += ["docx_markdown"] + cw
            if conf >= conf_threshold:
                return det_rows, "docx_tables_to_json", warnings, None

        try:
            rows2, usage = providers.universal_parse_rows(text)
            rows2 = explode_columnar_objects(rows2)
            return rows2, "docx_tables_to_json", warnings, usage
        except Exception as e:
            warnings.append(f"docx_llm_failed:{str(e)[:80]}")
            if det_rows:
                return det_rows, "docx_tables_to_json", warnings, None
            return [], "docx_tables_to_json", warnings, None

    if kind == "pdf":
        force_pdf_vision = os.getenv("FORCE_PDF_VISION", "1") == "1"
        ocr_rows: List[Dict[str, Any]] = []

        # OpenAI enabled: vision-first on PDFs is best for "recreate structure"
        if providers.openai_enabled and force_pdf_vision:
            md, u1 = providers.vision_to_markdown_from_pdf(data)
            warnings.append("pdf_vision_first")
            if md.strip():
                vis_rows = _parse_markdownish_tables(md)
                if vis_rows:
                    conf, cw = compute_confidence(vis_rows)
                    warnings += cw
                    if conf >= conf_threshold:
                        return vis_rows, "pdf_openai_vision_md_tables", warnings, u1
                    warnings.append("pdf_vision_low_confidence")
                if md.strip():
                    try:
                        rows, u2 = providers.universal_parse_rows(md)
                        rows = explode_columnar_objects(rows)
                        _merge_usage(u1, u2)
                        return rows, "pdf_openai_vision_md_to_json", warnings, u1
                    except Exception as e:
                        warnings.append(f"vision_failed:{str(e)[:100]}")
                        if vis_rows:
                            return vis_rows, "pdf_openai_vision_md_tables", warnings + ["pdf_vision_partial"], u1
            else:
                warnings.append("pdf_vision_empty")

        # Try direct table extraction first (best for structured PDFs)
        try:
            rows = _extract_pdf_tables_direct(data)
            if rows:
                conf, cw = compute_confidence(rows)
                warnings += ["pdf_table_direct"] + cw
                if conf >= conf_threshold:
                    return rows, "pdf_table_direct", warnings, usage
                warnings.append("pdf_table_direct_low_confidence")
        except Exception as e:
            warnings.append(f"pdf_table_direct_failed:{str(e)[:100]}")

        # Text extract attempt
        text = _extract_pdf_text(data)
        if text.strip():
            rows = _parse_markdown_table(text) or _parse_tsv_like(text) or _parse_csv_like(text)
            if not rows:
                try:
                    rows, u = providers.universal_parse_rows(text)
                    usage = u
                except Exception as e:
                    warnings.append(f"pdf_text_llm_failed:{str(e)[:80]}")
                    rows = []
            conf, cw = compute_confidence(rows)
            warnings += ["pdf_text_used"] + cw
            if conf >= conf_threshold:
                return rows, "pdf_text_to_json", warnings, usage
            warnings.append("pdf_text_low_confidence")

        # OCR attempt (if available)
        if OCR_AVAILABLE and PDF_RENDER_AVAILABLE:
            ocr_text = _ocr_pdf_first_pages(data, max_pages=int(os.getenv("PDF_OCR_PAGES", "2")))
            if ocr_text.strip():
                rows = _parse_markdown_table(ocr_text) or _parse_csv_like(ocr_text)
                if not rows:
                    try:
                        rows, u = providers.universal_parse_rows(ocr_text)
                        rows = explode_columnar_objects(rows)
                        usage = u
                    except Exception as e:
                        warnings.append(f"pdf_ocr_llm_failed:{str(e)[:80]}")
                        rows = []
                ocr_rows = rows
                conf, cw = compute_confidence(rows)
                warnings += ["pdf_ocr_used"] + cw
                if conf >= conf_threshold:
                    return rows, "pdf_ocr_to_json", warnings, usage
                warnings.append("pdf_ocr_low_confidence")
            else:
                warnings.append("pdf_ocr_empty")
        else:
            warnings.append("pdf_ocr_unavailable")

        # Vision fallback:
        if providers.openai_enabled:
            try:
                md, u1 = providers.vision_to_markdown_from_pdf(data)
                warnings.append("pdf_vision_fallback")
                md = (md or "").strip()
                if (not md) or md == "NO_TABLES":
                    warnings.append("pdf_vision_empty")
                elif md:
                    vis_rows = _parse_markdownish_tables(md)
                    if vis_rows:
                        conf, cw = compute_confidence(vis_rows)
                        warnings += cw
                        if conf >= conf_threshold:
                            return vis_rows, "pdf_openai_vision_md_tables", warnings, u1
                        warnings.append("pdf_vision_low_confidence")
                    if md.strip():
                        try:
                            rows, u2 = providers.universal_parse_rows(md)
                            rows = explode_columnar_objects(rows)
                            _merge_usage(u1, u2)
                            return rows, "pdf_openai_vision_md_to_json", warnings, u1
                        except Exception as e:
                            warnings.append(f"vision_failed:{str(e)[:100]}")
                            if vis_rows:
                                return vis_rows, "pdf_openai_vision_md_tables", warnings + ["pdf_vision_partial"], u1
            except Exception as e:
                warnings.append(f"vision_failed:{str(e)[:100]}")
                if ocr_rows:
                    return ocr_rows, "pdf_ocr_to_json_no_vision", warnings + ["pdf_vision_failed"], usage

        # Non-OpenAI vision: render pages -> vision on images
        if not PDF_RENDER_AVAILABLE:
            raise RuntimeError("pdf_render_unavailable_for_non_openai_vision (install pdf2image+poppler)")

        pages = convert_from_bytes(
            data,
            dpi=int(os.getenv("PDF_VISION_DPI", "220")),
            first_page=1,
            last_page=int(os.getenv("PDF_VISION_PAGES", "2")),
        )
        mds: List[str] = []
        for img in pages:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            md, u = providers.vision_to_markdown_from_image(buf.getvalue())
            usage = u  # last usage
            mds.append(md)

        md_all = "\n\n".join([m.strip() for m in mds if m.strip()])
        md_all = (md_all or "").strip()
        if (not md_all) or md_all == "NO_TABLES":
            warnings.append("pdf_vision_empty")
            if ocr_rows:
                return ocr_rows, "pdf_ocr_to_json_no_vision", warnings + ["no_vision_using_ocr_low_conf"], usage
            return [], "pdf_rendered_pages_vision_empty", warnings, usage

        warnings.append("pdf_vision_non_openai")
        rows2 = _parse_markdownish_tables(md_all)
        if rows2:
            conf, cw = compute_confidence(rows2)
            warnings += cw
            if conf >= conf_threshold:
                return rows2, "pdf_render_vision_md_tables", warnings, usage
            warnings.append("pdf_vision_low_confidence")
        if not (md_all or "").strip():
            if ocr_rows:
                return ocr_rows, "pdf_render_vision_empty_returning_ocr", warnings + ["pdf_render_vision_empty"], usage
            return [], "pdf_render_vision_empty", warnings + ["pdf_render_vision_empty"], usage
        try:
            rows2, u2 = providers.universal_parse_rows(md_all)
            usage = usage or u2
            return rows2, "pdf_render_vision_md_to_json", warnings, usage
        except Exception as e:
            warnings.append(f"vision_failed:{str(e)[:100]}")
            if ocr_rows:
                return ocr_rows, "pdf_ocr_to_json_no_vision", warnings + ["pdf_vision_failed"], usage
            return [], "pdf_render_vision_md_to_json", warnings, usage

    if kind == "image":
        # OCR attempt
        if OCR_AVAILABLE:
            ocr = _ocr_image(data)
            if ocr.strip():
                # Try deterministic parsers first (including whitespace-separated tables)
                rows = (_parse_markdown_table(ocr) or
                       _parse_tsv_like(ocr) or
                       _parse_csv_like(ocr) or
                       _parse_whitespace_table(ocr))

                if not rows:
                    # Try LLM parsing only if available
                    if providers.openai_enabled or providers.hf_enabled:
                        try:
                            rows, u = providers.universal_parse_rows(ocr)
                            rows = explode_columnar_objects(rows)
                            usage = u
                        except Exception as e:
                            warnings.append(f"image_ocr_llm_failed:{str(e)[:80]}")
                            rows = []
                    else:
                        # No LLM available, but we have OCR text - return empty with warning
                        warnings.append("ocr_extracted_but_no_llm_to_parse")
                        rows = []

                if rows:
                    conf, cw = compute_confidence(rows)
                    warnings += ["image_ocr_used"] + cw
                    if conf >= conf_threshold:
                        return rows, "image_ocr_to_json", warnings, usage
                    warnings.append("image_ocr_low_confidence")
                    # Save OCR rows in case vision isn't available
                    ocr_rows = rows
            else:
                warnings.append("image_ocr_empty")
        else:
            warnings.append("image_ocr_unavailable")

        # Vision fallback (only if LLM provider available and working)
        # Try vision, but catch errors if provider isn't actually available
        try:
            if providers.openai_enabled or providers.hf_enabled:
                md, u1 = providers.vision_to_markdown_from_image(data)
                warnings.append("image_vision_used")
                vis_rows = _parse_markdownish_tables(md)
                if vis_rows:
                    conf, cw = compute_confidence(vis_rows)
                    warnings += cw
                    if conf >= conf_threshold:
                        return vis_rows, "image_vision_md_tables", warnings, u1
                    warnings.append("image_vision_low_confidence")
                try:
                    rows, u2 = providers.universal_parse_rows(md)
                    rows = explode_columnar_objects(rows)
                    _merge_usage(u1, u2)
                    return rows, "image_vision_md_to_json", warnings, u1
                except Exception as e:
                    warnings.append(f"image_vision_llm_failed:{str(e)[:100]}")
        except Exception as e:
            # Vision failed - fall through to use OCR results
            warnings.append(f"vision_failed:{str(e)[:100]}")

        # No vision available or vision failed - return OCR results if we have them, even with low confidence
        if 'ocr_rows' in locals() and ocr_rows:
            warnings.append("no_vision_using_ocr_low_conf")
            return ocr_rows, "image_ocr_to_json_no_vision", warnings, usage

        # No OCR or vision available
        warnings.append("no_ocr_no_vision_available")
        return [], "image_no_extraction", warnings, usage

    return [], "unsupported", warnings + [f"unsupported_kind={kind}"], usage


def _merge_usage(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    # best-effort: sum if both openai
    if not a or not b:
        return
    if a.get("provider") == "openai" and b.get("provider") == "openai":
        a["input_tokens"] = int(a.get("input_tokens", 0)) + int(b.get("input_tokens", 0))
        a["output_tokens"] = int(a.get("output_tokens", 0)) + int(b.get("output_tokens", 0))


# Parsing helpers
def _parse_markdownish_tables(text: str) -> List[Dict[str, Any]]:
    """Try deterministic markdown/tsv/whitespace/csv table parsing before LLM."""
    return (
        _parse_markdown_table(text)
        or _parse_tsv_like(text)
        or _parse_whitespace_table(text)
        or _parse_csv_like(text)
    )

def _parse_markdown_table(text: str) -> List[Dict[str, Any]]:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    for i in range(len(lines) - 1):
        if "|" in lines[i] and re.search(r"\|\s*-", lines[i + 1]):
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            out: List[Dict[str, Any]] = []
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if len(row) == len(header):
                    out.append(dict(zip(header, row)))
                j += 1
            return out
    return []


def _parse_csv_like(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    if not text or "\n" not in text:
        return []
    first = text.splitlines()[0]
    if first.count(",") < 2:
        return []
    try:
        reader = csv.DictReader(io.StringIO(text))
        return [dict(r) for r in reader]
    except Exception:
        return []


def _parse_whitespace_table(text: str) -> List[Dict[str, Any]]:
    """
    Parse space/whitespace-separated tables from OCR output.
    Handles tables like:
        Lane POL POD Type Rate Transit FDO FDD
        L020 INMIAA NLRTM 40HC $3200 22days 10 11
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    # Find the header line - must contain POL and POD (shipping table indicators)
    header_idx = -1
    for i, line in enumerate(lines):
        # Look for shipping-specific header patterns
        lower = line.lower()
        # Must contain both POL and POD (key shipping fields)
        if 'pol' in lower and 'pod' in lower:
            # Additional validation - should have Lane or Type nearby
            if 'lane' in lower or 'type' in lower:
                header_idx = i
                break

    if header_idx == -1:
        return []

    # Parse header - split by whitespace
    header_line = lines[header_idx]
    headers = header_line.split()

    if len(headers) < 4:
        return []

    # Parse data rows
    rows: List[Dict[str, Any]] = []
    for line in lines[header_idx + 1:]:
        # Skip empty lines, notes, comments, or lines that don't look like data
        if (not line or
            line.startswith('-') or
            line.startswith('Note') or
            'same as' in line.lower() or
            'accept' in line.lower() or
            'sailing' in line.lower() or
            'container count' in line.lower() or
            len(line) < 10):
            continue

        # Split by whitespace
        parts = line.split()

        # Data rows should start with a lane ID (like L020, L021, etc.)
        if not parts or not parts[0].startswith('L'):
            continue

        # Need at least half the headers filled (be lenient with OCR errors)
        if len(parts) < len(headers) // 2:
            continue

        # Map to headers - pad with empty strings if needed
        row_dict = {}
        for i, header in enumerate(headers):
            if i < len(parts):
                row_dict[header] = parts[i]
            else:
                row_dict[header] = ""  # Missing value

        # Only add if we have meaningful data (lane ID and at least POL/POD)
        if row_dict.get('Lane', '').startswith('L') and (row_dict.get('POL') or row_dict.get('POD')):
            rows.append(row_dict)

    return rows



# Extractors
def _extract_excel(buf: bytes) -> List[Dict[str, Any]]:
    # requires openpyxl installed
    xls = pd.ExcelFile(io.BytesIO(buf))
    out: List[Dict[str, Any]] = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        if df is None or df.empty:
            continue
        df = df.dropna(how="all")
        for rec in df.to_dict(orient="records"):
            if any(str(v).strip() not in ("", "nan", "None") for v in rec.values()):
                out.append(rec)
    return out


def _extract_csv(buf: bytes) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(buf))
    return df.to_dict(orient="records")


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        parts: List[str] = []
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts).strip()


def _extract_pdf_tables_direct(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract tables directly from PDF using pdfplumber.
    Handles cases where tables are split across multiple detected table regions.
    """
    def clean_header(h: str) -> str:
        """Remove newlines and extra spaces from headers"""
        return (h or "").replace("\n", "").strip()

    def clean_value(v: str) -> str:
        """Clean cell values - remove newlines and extra spaces"""
        return (v or "").replace("\n", " ").strip()

    all_rows: List[Dict[str, Any]] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            if not tables:
                continue

            # Check if tables come in pairs (split rows)
            # This handles PDFs where one logical row is split into multiple table regions
            if len(tables) >= 2:
                # Try to detect if tables are paired (same number of data rows)
                paired = True
                for i in range(0, len(tables) - 1, 2):
                    if i + 1 < len(tables):
                        t1, t2 = tables[i], tables[i + 1]
                        if not (t1 and t2 and len(t1) > 1 and len(t2) > 1):
                            paired = False
                            break
                        if len(t1) != len(t2):
                            paired = False
                            break

                if paired:
                    # Merge paired tables
                    for i in range(0, len(tables), 2):
                        if i + 1 < len(tables):
                            t1, t2 = tables[i], tables[i + 1]
                            if t1 and t2 and len(t1) > 1 and len(t2) > 1:
                                # Combine headers
                                headers1 = [clean_header(h) for h in t1[0]]
                                headers2 = [clean_header(h) for h in t2[0]]
                                combined_headers = headers1 + headers2

                                # Combine data rows
                                for row_idx in range(1, len(t1)):
                                    data1 = t1[row_idx]
                                    data2 = t2[row_idx] if row_idx < len(t2) else []
                                    combined_data = [clean_value(v) for v in data1] + [clean_value(v) for v in data2]

                                    if len(combined_data) == len(combined_headers):
                                        row_dict = dict(zip(combined_headers, combined_data))
                                        # Only add if not all empty
                                        if any(str(v).strip() for v in row_dict.values()):
                                            all_rows.append(row_dict)
                    return all_rows

            # Fallback: process tables independently
            for table in tables:
                if not table or len(table) < 2:
                    continue

                headers = [clean_header(h) for h in table[0]]
                for row in table[1:]:
                    if len(row) == len(headers):
                        row_dict = dict(zip(headers, [clean_value(v) for v in row]))
                        if any(str(v).strip() for v in row_dict.values()):
                            all_rows.append(row_dict)

    return all_rows


def _ocr_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return (pytesseract.image_to_string(img) or "").strip() if pytesseract else ""


def _ocr_pdf_first_pages(pdf_bytes: bytes, max_pages: int = 2) -> str:
    if not (PDF_RENDER_AVAILABLE and pytesseract):
        return ""
    pages = convert_from_bytes(pdf_bytes, dpi=220, first_page=1, last_page=max_pages)
    texts: List[str] = []
    for img in pages:
        texts.append(pytesseract.image_to_string(img) or "")
    return "\n\n".join([t.strip() for t in texts if t.strip()])


def _extract_docx_tables_to_text(docx_bytes: bytes) -> str:
    assert Document is not None
    doc = Document(io.BytesIO(docx_bytes))

    blocks: List[str] = []
    for t_i, table in enumerate(doc.tables, 1):
        blocks.append(f"Table {t_i}")
        for row_idx, row in enumerate(table.rows):
            cells = [c.text.strip() for c in row.cells]
            blocks.append(" | ".join(cells))
            if row_idx == 0 and cells:
                blocks.append(" | ".join(["---"] * len(cells)))
        blocks.append("")  # blank line between tables
    return "\n".join(blocks).strip()


# --------------------------
# Normalization + confidence + schema validation
# --------------------------

def normalize_row_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    keymap = {
        "pol_requested": "requested_POL",
        "pod_requested": "requested_POD",
        "pol_quoted": "quoted_POL",
        "pod_quoted": "quoted_POD",
        "of_rate": "of_rate_usd",
        "ocean_freight": "of_rate_usd",
        "ocean_freight_rate": "of_rate_usd",
        "container_cnt": "container_count",
        "transit_time": "transit_time_days",
        "free_days_origin_days": "free_days_origin",
        "free_days_destination_days": "free_days_destination",
    }
    keymap.update({
        "lane": "lane_id",
        "lane_id": "lane_id",

        "pol": "requested_POL",
        "pod": "requested_POD",
        "requested_pol": "requested_POL",
        "requested_pod": "requested_POD",
        "quoted_pol": "quoted_POL",
        "quoted_pod": "quoted_POD",

        "container": "container_type",
        "type": "container_type",
        "container_coumt": "container_count",  # Handle OCR typo

        "count": "container_count",
        "container_count": "container_count",

        "transit": "transit_time_days",
        "transit_time_days": "transit_time_days",

        "frequency": "sailing_frequency",
        "sailing_frequency": "sailing_frequency",

        "rate_usd": "of_rate_usd",
        "rate_usd_": "of_rate_usd",
        "rate": "of_rate_usd",

        "fd_orig": "free_days_origin",
        "fd_origin": "free_days_origin",
        "free_days_origin": "free_days_origin",

        "fd_dest": "free_days_destination",
        "fd_destination": "free_days_destination",
        "free_days_destination": "free_days_destination",

        "ops_req": "accept_operational_requirements",
        "accept_operational_requirements": "accept_operational_requirements",

        "payment": "accept_payment_methods",
        "accept_payment_methods": "accept_payment_methods",

        "fhnd": "FND",  # Handle OCR typo FHND -> FND
        "fnd": "FND",
    })

    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        kk = str(k).strip()
        canon = re.sub(r"[^a-z0-9]+", "_", kk.lower()).strip("_")
        target = keymap.get(canon, kk)
        out[target] = _to_number(v)
    return out


def _to_number(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v).strip()
    if s == "":
        return ""
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False

    # Strip common non-numeric characters (currency symbols, units, etc.)
    s2 = s.replace(",", "")
    s2 = s2.replace("$", "").replace("â‚¬", "").replace("Â£", "")
    s2 = s2.replace(" days", "").replace(" day", "")
    s2 = s2.replace(" hours", "").replace(" hour", "")
    s2 = s2.strip()

    try:
        if re.fullmatch(r"-?\d+", s2):
            return int(s2)
        if re.fullmatch(r"-?\d+\.\d+", s2):
            return float(s2)
    except Exception:
        return v
    return v


def compute_confidence(rows: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
    if not rows:
        return 0.0, ["no_rows"]

    must = ["lane_id", "requested_POL", "requested_POD", "container_type", "of_rate_usd"]
    scores: List[float] = []
    for r in rows:
        s = 0.0
        present = sum(1 for f in must if str(r.get(f) or "").strip() != "")
        s += present / len(must)

        pol = str(r.get("requested_POL") or "").strip().upper()
        pod = str(r.get("requested_POD") or "").strip().upper()
        if pol and PORT_LIKE_RE.match(pol):
            s += 0.15
        if pod and PORT_LIKE_RE.match(pod):
            s += 0.15

        rate = r.get("of_rate_usd")
        if isinstance(rate, (int, float)) and rate > 0:
            s += 0.2

        scores.append(min(s, 1.0))

    conf = float(statistics.mean(scores))
    warns: List[str] = []
    if conf < 0.5:
        warns.append(f"low_confidence={conf:.2f}")
    return conf, warns


def validate_and_clean_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """
    Enforces:
      - required fields exist (or are derivable)
      - correct-ish types (coerce)
      - UN/LOCODE-like validation for ports (requested/quoted POL/POD, POR, FND)
    Returns: (valid_rows, warnings, rejects[{row, why}])
    """
    warnings: List[str] = []
    rejects: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []

    for r in rows:
        rr = dict(r)

        # Fill quoted_* from requested_* if missing
        if not rr.get("quoted_POL") and rr.get("requested_POL"):
            rr["quoted_POL"] = rr.get("requested_POL")
        if not rr.get("quoted_POD") and rr.get("requested_POD"):
            rr["quoted_POD"] = rr.get("requested_POD")

        # Coerce types
        for f in ("container_count", "transit_time_days", "free_days_origin", "free_days_destination", "of_rate_usd"):
            if f in rr:
                rr[f] = _to_number(rr[f])
        for f in ("accept_operational_requirements", "accept_payment_methods"):
            if f in rr:
                rr[f] = _to_number(rr[f])

        # Basic required fields check
        missing = [f for f in REQUIRED_FIELDS if rr.get(f) in (None, "", [])]
        # allow some fields to be absent in raw data but still pass (weâ€™ll keep them empty)
        hard_required = ["lane_id", "requested_POL", "requested_POD", "container_type", "of_rate_usd"]
        hard_missing = [f for f in hard_required if rr.get(f) in (None, "", [])]

        # UN/LOCODE-ish validation for ports (only when present)
        port_fields = ["POR", "requested_POL", "quoted_POL", "requested_POD", "quoted_POD", "FND"]
        bad_ports = []
        for pf in port_fields:
            v = rr.get(pf)
            if v in (None, "", []):
                continue
            sv = str(v).strip().upper()
            if not PORT_LIKE_RE.match(sv):
                bad_ports.append((pf, sv))

        if bad_ports:
            warnings.append(f"non_unlocode_ports={len(bad_ports)}")

        if hard_missing:
            rejects.append({"row": rr, "why": f"missing_hard_required={hard_missing}"})
            continue

        # Numeric sanity
        if isinstance(rr.get("of_rate_usd"), (int, float)) and rr["of_rate_usd"] <= 0:
            rejects.append({"row": rr, "why": "of_rate_usd_nonpositive"})
            continue

        valid.append(rr)

    return valid, warnings, rejects