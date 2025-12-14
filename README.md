# Ocean Freight Quote Extraction System

Automated extraction and normalization of shipping quotes from carrier emails. Handles multiple formats (PDF, Excel, CSV, images, Word docs) using a hybrid approach combining deterministic parsing, LLMs, and vision models.

## What It Does

Takes carrier quote emails with various attachments and outputs:
- Normalized CSV/Parquet dataset with standardized fields
- Processing report showing extraction methods and confidence
- Auto-generated negotiation email drafts per carrier

**Performance:** For the included 4-carrier pack, the run completes in ~4m36s with estimated OpenAI cost ~$0.0011 (gpt-5-mini + gpt-5-nano).

**Test Data:** The repository includes 4 original assignment emails plus 6 additional test cases (extra) for comprehensive validation (10 carriers total).


## Table of Contents

- [Setup](#setup)
- [Running the System](#running-the-system)
- [Configuration](#configuration)
- [System Architecture](#system-architecture)
- [Future Roadmap](#future-roadmap)
- [Troubleshooting](#troubleshooting)

***

## Setup

### Prerequisites

- **Python 3.11+**
- **Docker** (optional but recommended)
- **OpenAI API key** (required for default configuration)
- **HuggingFace token** (optional, for free Qwen vision model)

### Getting API Keys

#### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/signup)
2. Sign up or log in
3. Navigate to [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Copy the key (format: `sk-...`)
6. **Important**: Add billing information to your OpenAI account

#### HuggingFace Token (Optional)
1. Visit [HuggingFace](https://huggingface.co/join)
2. Sign up or log in
3. Navigate to [Access Tokens](https://huggingface.co/settings/tokens)
4. Click "New token"
5. Select "Read" access type
6. Copy the token (format: `hf_...`)

**Free tier**: HuggingFace inference API has rate limits but is free for testing.

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/ahsham/ocean-freight-extractor.git
cd ocean-freight-extractor
```

#### 2. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```bash
# Required for default setup
OPENAI_API_KEY=sk-your-actual-key-here

# Optional: free alternative vision provider
HF_TOKEN=hf_your-actual-token-here
```

***

## Running the System

### Option 1: Docker (Recommended)

**Why Docker?**
- Consistent environment across platforms
- Pre-installed system dependencies (poppler, tesseract)
- No Python version conflicts

#### Build the Image
```bash
docker build -t quote-extractor .
```

#### Run the Container

**Linux/Mac:**
```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/input:/in" \
  -v "$(pwd)/output:/out" \
  quote-extractor \
  --input /in --output /out
```

**Windows PowerShell:**
```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/input:/in" `
  -v "${PWD}/output:/out" `
  quote-extractor `
  --input /in --output /out
```

**Windows CMD:**
```cmd
docker run --rm ^
  --env-file .env ^
  -v "%cd%/input:/in" ^
  -v "%cd%/output:/out" ^
  quote-extractor ^
  --input /in --output /out
```

#### Alternative: Pass Environment Variables Directly
```bash
docker run --rm \
  -e OPENAI_API_KEY=sk-your-key \
  -e HF_TOKEN=hf_your-token \
  -v "$(pwd)/input:/in" \
  -v "$(pwd)/output:/out" \
  quote-extractor \
  --input /in --output /out
```

#### Check Container Logs
```bash
# Real-time logs
docker run --rm --env-file .env -v "$(pwd)/input:/in" -v "$(pwd)/output:/out" quote-extractor | tee run.log

# Or check output/run.log after completion
tail -f output/run.log
```

### Option 2: Local Python

**Install System Dependencies:**

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

**macOS:**
```bash
brew install poppler tesseract
```

**Windows:**
- Poppler: Download from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
- Tesseract: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Add both to your system PATH

**Install Python Dependencies:**

**With pip:**
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --input input --output output

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --input input --output output
```

**With conda:**
```bash
conda create -n quote-extractor python=3.11
conda activate quote-extractor
pip install -r requirements.txt
python main.py --input input --output output
```

### Input/Output

**Input:** Place `.eml` files in `input/` directory
- Supported attachments: PDF, Excel (.xlsx), CSV, Images (.png/.jpg), Word (.docx)

**Output:** Results written to `output/` directory
- `normalized_quotes.csv` / `.parquet` - Standardized data
- `run_report.json` - Extraction metadata
- `review_items.json` - Flagged items (validation failures)
- `run.log` - Processing logs
- `drafts/*.md` - Negotiation emails per carrier

***

## Configuration

### Environment Variables

```bash
# ===== OpenAI Configuration =====
OPENAI_ENABLED=1
OPENAI_API_KEY=sk-your-key
OPENAI_VISION_MODEL=gpt-5-mini-2025-08-07
OPENAI_TEXT_MODEL=gpt-5-nano-2025-08-07

# OpenAI Pricing (per million tokens)
OPENAI_VISION_INPUT_PER_M=0.05
OPENAI_VISION_OUTPUT_PER_M=0.40
OPENAI_TEXT_INPUT_PER_M=0.05
OPENAI_TEXT_OUTPUT_PER_M=0.40

# ===== HuggingFace Configuration =====
HF_TOKEN=hf_your-token
HF_VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct

# ===== Local Transformers (Future) =====
# HF_LOCAL_VL_MODEL=/path/to/local/model
# HF_LOCAL_VL_DEVICE=cuda  # or cpu

# ===== Processing Options =====
FORCE_PDF_VISION=0           # 1 = always use vision for PDFs (slower, better for images)
CONF_THRESHOLD=0.55          # Minimum confidence for auto-approval (0.0-1.0)
PDF_OCR_PAGES=2              # Max pages for OCR fallback
PDF_VISION_PAGES=2           # Max pages for vision extraction
PDF_VISION_DPI=220           # DPI for PDF-to-image rendering

# ===== Debugging =====
DEBUG=0                      # 1 = enable verbose logging
```

### Command-Line Options

```bash
python main.py \
  --input /path/to/emails \
  --output /path/to/results \
  --conf_threshold 0.6 \
  --debug
```

### Cost Optimization Strategies

1. **Default (Minimal Cost):** GPT-5 mini/nano (~$0.002 per 10 carriers)
2. **Zero Cost:** Use HuggingFace Qwen (free, rate-limited)
3. **Best Quality:** GPT-4o for vision extraction (~$0.05 per run)
4. **Future:** Fine-tuned local models (see [Future Roadmap](#future-roadmap))

***

## System Architecture

### Design Philosophy

**Hybrid Extraction Pipeline:** Combines deterministic parsing, LLM intelligence, and vision models to handle diverse input formats with high accuracy and low cost.

### Architecture Diagram

```
┌──────────────┐
│  Email (.eml)│
│  + Attachments│
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Format Detection & Routing         │
├─────────────────────────────────────┤
│  • PDF → Text/Image classification  │
│  • Excel/CSV → Direct parsing       │
│  • Images → OCR/Vision pipeline     │
│  • Email body → Pattern matching    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Extraction Layer (Cascading)       │
├─────────────────────────────────────┤
│  1. Deterministic parsers (fastest) │
│  2. LLM structured extraction       │
│  3. Vision models (fallback)        │
│  4. OCR + LLM (last resort)         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Normalization & Validation         │
├─────────────────────────────────────┤
│  • Field name mapping               │
│  • Type coercion (str → float)      │
│  • UNLOCODE validation              │
│  • Confidence scoring               │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Negotiation Engine                 │
├─────────────────────────────────────┤
│  • Market analysis (spot estimates) │
│  • Target calculation               │
│  • LLM-generated emails             │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Output Layer                       │
│  CSV/Parquet + Reports + Drafts     │
└─────────────────────────────────────┘
```

### Key Components

#### 1. **Extractors** (`extractors.py`)
- **Purpose**: Format-specific data extraction
- **Strategy**: Cascading fallbacks (deterministic → LLM → vision)
- **Highlights**:
  - Handles split-table PDFs (multi-region table detection)
  - OCR fallback for scanned documents
  - Confidence scoring based on field completeness

#### 2. **LLM Providers** (`llm_providers.py`)
- **Purpose**: Unified interface for OpenAI & HuggingFace
- **Features**:
  - Automatic vision fallback (OpenAI → HuggingFace if empty)
  - Token usage tracking & cost estimation
  - Reasoning model detection (GPT-5, o1)

#### 3. **Negotiation Engine** (`negotiate.py`)
- **Purpose**: Generate data-driven email responses
- **Inputs**: Normalized quotes + market rules (spot ±5%, prior +30%)
- **Output**: Per-carrier markdown emails with:
  - Target counteroffers
  - Justification (market/peer/historical comparisons)
  - Give/gets (e.g., "accept ops requirements for 11 free days")

#### 4. **Orchestrator** (`main.py`)
- **Purpose**: End-to-end workflow coordination
- **Logging**: Structured JSON + human-readable logs
- **Error Handling**: Per-file isolation (one failure doesn't stop batch)

### Prompt Engineering & Guardrails

#### Universal Table Extraction Prompt
```
You are an expert data extraction AI. Extract quote data from tables.
Output ONLY valid JSON array. No markdown, no commentary.
Booleans: true/false. Numbers: numeric type. Preserve case.
```

**Guardrails:**
- Response format validation (JSON-only extraction)
- Retry logic (2 attempts with exponential backoff)
- Empty response handling (triggers fallback extraction)

#### Vision-to-Markdown Prompt
```
Extract ALL tables from this document.
Output ONLY GitHub-flavored Markdown tables.
If NO tables found, output exactly: "NO_TABLES"
```

**Guardrails:**
- Structured output enforcement
- Fallback to OCR if vision returns empty
- DPI/page limits to control costs

#### Negotiation Email Prompt
```
You are a Senior Procurement Manager at Cargoo.
Draft a professional negotiation email with:
- Data-driven counteroffers (cite spot/peer/historical)
- Clear give/gets
- Actionable next steps

Tone: Professional, firm, collaborative.
Use ONLY the provided data. Do NOT invent numbers.
```

**Guardrails:**
- JSON data injection (prevents hallucination)
- Temperature=0.2 (consistency over creativity)
- Field validation (UNLOCODE format checks)

### Error Handling Strategy

1. **Per-File Isolation**: One email failure doesn't crash batch
2. **Graceful Degradation**:
   - Vision fails → OCR fallback
   - OCR fails → Skip with warning
   - LLM fails → Return deterministic parse (lower confidence)
3. **Review Queue**: Invalid rows → `review_items.json` for human review
4. **Logging**: DEBUG mode captures full LLM responses

### Observability

**Metrics Tracked:**
- Extraction method per file (deterministic/LLM/vision)
- Confidence scores (0.0-1.0)
- Token usage (input/output/cost)
- Processing time per carrier
- Validation failures

**Log Levels:**
- INFO: High-level progress
- DEBUG: LLM prompts, responses, intermediate parsing
- WARNING: Fallbacks triggered, validation issues
- ERROR: Unhandled exceptions

**Output Artifacts:**
- `run_report.json`: Structured metadata (JSON schema)
- `review_items.json`: Flagged items with reasons
- `run.log`: Timestamped event log

***

## Future Roadmap (Assuming 1 person working with minimal budget)

### Phase 1: Production Hardening (expected Quarter 1, 2026)
- [ ] Add retry logic with exponential backoff for API failures
- [ ] Implement streaming for large PDFs (memory optimization)
- [ ] Add webhook support for async processing
- [ ] Create web UI for upload + review queue management

### Phase 2: Local Models (expected Quarter 2 2026)

#### Why Local Models?
- **Cost**: Eliminate per-token API fees
- **Privacy**: Keep sensitive shipping data on-premises
- **Latency**: <100ms inference vs. 2-5s API calls
- **Reliability**: No dependency on external services

#### Implementation Plan

**1. Download & Cache Models**
```bash
# Download Qwen2.5-VL-7B-Instruct/Qwen3-VL-8B-Instruct locally
pip install transformers torch torchvision
python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
model_name = 'Qwen/Qwen2-VL-7B-Instruct'
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name, device_map='auto')
print('Model downloaded to ~/.cache/huggingface')
"
```

**2. Configure Local Inference**
```bash
# .env
HF_LOCAL_VL_MODEL=Qwen/Qwen2-VL-7B-Instruct
HF_LOCAL_VL_DEVICE=cuda  # or cpu
OPENAI_ENABLED=0  # Disable API calls
```

**3. Hardware Requirements**
- **GPU**: NVIDIA with 16GB+ VRAM (RTX 4090, A10, etc.)
- **CPU**: 32GB+ RAM for CPU-only inference (slower)
- **Storage**: ~15GB per model

**Current Support:** `llm_providers.py` already has `hf_local_image_to_md()` method for local vision models.

### Phase 3: Fine-Tuning for Domain Adaptation (expected Quarter 3 2026)

#### Why Fine-Tune?
- **Accuracy**: Specialized models outperform generalists on niche tasks
- **Speed**: Smaller fine-tuned models (1-3B params) can match GPT-4 on-domain
- **Cost**: Fine-tuned Llama3-8B inference is ~10x cheaper than GPT-4o API

#### Fine-Tuning Strategy

**Target Tasks:**
1. **Table Extraction**: PDF/Image → Structured JSON
2. **Field Normalization**: "of rate" → "ofrateusd", "Transit" → "transittimedays"
3. **Negotiation Generation**: Quotes + market data → Email draft

**Approach: LoRA (Low-Rank Adaptation)**
- Fine-tune only 0.1% of model parameters
- Trains on 1x RTX 4090 in ~2 hours
- Preserves base model knowledge

#### Dataset Creation

**Step 1: Synthetic Data Generation**
```python
# generate_training_data.py
import random
from fpdf import FPDF

def create_synthetic_quote():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    lanes = [("CNSHA", "NLRTM"), ("SGSIN", "USNYC"), ...]
    rates = [2000 + random.randint(-500, 500) for _ in lanes]

    # Render table with variations (different fonts, layouts, OCR noise)
    for (pol, pod), rate in zip(lanes, rates):
        pdf.cell(200, 10, txt=f"{pol} → {pod} | ${rate} | 20 days", ln=True)

    return pdf.output(dest='S').encode('latin-1')

# Generate 10,000 synthetic quote PDFs with ground truth labels
```

**Step 2: Real Data Annotation**
- Use current system to process real emails
- Human review flags corrections
- Build dataset: {input: PDF bytes, output: normalized JSON}

**Step 3: Fine-Tuning Script**
```bash
# Install training framework
pip install peft transformers bitsandbytes

# Run fine-tuning (example with Qwen2-VL)
python finetune_vision_model.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset synthetic_quotes_10k.jsonl \
  --lora_rank 16 \
  --epochs 3 \
  --output_dir ./models/qwen2-vl-quotes-finetuned
```

**Expected Improvements:**
- Extraction accuracy: high on provided pack (all 12 rows validated); extendable with review queue.
- Inference speed: 50% faster (smaller model)
- API costs: $0.002/run → $0 (after initial training investment)

### Phase 4: Advanced Features (expected Quarter 4 2026)
- [ ] Multi-language support (Chinese, Japanese shipping docs)
- [ ] Anomaly detection (flag unusual rates/terms)
- [ ] RAG integration (retrieve historical tender data for better negotiation)
- [ ] Auto-approval workflow (high-confidence quotes → direct award)

### Research Directions
- **Test-Time Compute**: Use reasoning models (o1, DeepSeek-R1) for complex negotiations
- **Multimodal Fusion**: Combine email text + attachment content for better context
- **Active Learning**: System suggests which low-confidence items need human review first

***

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY missing"**
```bash
# Check if .env file exists
cat .env | grep OPENAI_API_KEY

# Verify key is loaded
docker run --rm --env-file .env alpine printenv | grep OPENAI
```

**Parquet file not created**
```bash
pip install pyarrow
# Or in Dockerfile: add pyarrow to requirements.txt
```

**"('Could not convert Transit' with type str")**
- Issue: Non-numeric value in numeric column
- Fix: Already handled in latest code (see `clean_for_parquet()`)
- Verify: Check `output/normalized_quotes.csv` for "Transit" strings in data rows

**HuggingFace rate limit errors**
```bash
# Switch to OpenAI or reduce concurrent requests
OPENAI_ENABLED=1
# Or: wait 1 hour (free tier resets hourly)
```

**Docker: "Cannot connect to the Docker daemon"**
```bash
# Linux: Start Docker service
sudo systemctl start docker

# Windows/Mac: Ensure Docker Desktop is running
```

**Vision extraction returns empty**
- **Cause**: Low-quality images, unsupported layouts
- **Fix**: Enable force-vision mode or increase DPI
```bash
FORCE_PDF_VISION=1
PDF_VISION_DPI=300
```

### Platform-Specific

**Linux (Ubuntu/Debian)**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip poppler-utils tesseract-ocr

# Python 3.11 not available? Use deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
```

**macOS**
```bash
# Install dependencies
brew install python@3.11 poppler tesseract

# Link Python 3.11
brew link python@3.11
```

**Windows**
- Use PowerShell (not Git Bash for Docker)
- Add poppler/tesseract to PATH
- For Docker: Enable WSL2 backend in Docker Desktop settings

### Debugging

**Enable verbose logging:**
```bash
DEBUG=1 #in the .env file
python main.py --input input --output output
```

**Check intermediate files:**
```bash
# LLM responses are logged in DEBUG mode
tail -f output/run.log | grep "LLM response"
```

**Test individual components:**
```python
# Test PDF extraction
from extractors import extract_pdf_tables_direct
with open("test.pdf", "rb") as f:
    rows = extract_pdf_tables_direct(f.read())
print(rows)
```

***

## Project Structure

```
.
├── main.py                 # Orchestrator (email parsing → output)
├── extractors.py           # Format-specific extraction logic
├── llm_providers.py        # OpenAI + HuggingFace integrations
├── negotiate.py            # Negotiation email generation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── .env.example            # Environment template
├── .env                    # Your secrets (git-ignored)
├── .gitignore
├── .dockerignore
├── README.md               # This file
├── data_ref/               # UN/LOCODE port database (future work)
├── input/                  # Place .eml files here
│   ├── email_carrier_alpha.eml
│   ├── email_carrier_beta.eml
│   └── ...
└── output/                 # Results directory
    ├── normalized_quotes.csv
    ├── normalized_quotes.parquet
    ├── run_report.json
    ├── review_items.json
    ├── run.log
    └── drafts/
        ├── Carrier_Alpha.md
        └── ...
```

***

## Performance Benchmarks

**Test Configuration:** 10 carriers, 32 lanes total

| Metric | Value |
|--------|-------|
| Total processing time | ~4 minutes |
| OpenAI API cost | $0.002 |
| Extraction accuracy | 95%+ (manual validation) |
| Memory usage | <500MB |
| Docker image size | 450MB |

***

## Schema Reference

**Normalized Output Fields:**

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| carrier | string | "Carrier Alpha" | Carrier name |
| source | string | "email.eml/attachment.pdf" | Origin file |
| extraction_method | string | "pdf_vision_md_tables" | Method used |
| confidence | float | 1.0 | Confidence score (0-1) |
| lane_id | string | "L001" | Lane identifier |
| POR | string | "CNHGH" | Place of receipt |
| requested_POL | string | "CNSHA" | Requested port of loading |
| quoted_POL | string | "CNSHA" | Quoted port of loading |
| requested_POD | string | "NLRTM" | Requested port of discharge |
| quoted_POD | string | "NLRTM" | Quoted port of discharge |
| FND | string | "BEANR" | Final destination |
| container_type | string | "40HC" | Container type |
| container_count | int | 10 | Number of containers |
| accept_operational_requirements | bool | true | Ops requirements accepted |
| accept_payment_methods | bool | true | Payment terms accepted |
| transit_time_days | float | 27.0 | Transit time (days) |
| sailing_frequency | string | "Every 10 days" | Sailing frequency |
| of_rate_usd | float | 3535.0 | Ocean freight rate (USD) |
| free_days_origin | float | 7.0 | Free days at origin |
| free_days_destination | float | 14.0 | Free days at destination |

***

## License

All rights reserved. This code is proprietary and may not be copied, distributed, or modified without explicit permission.

***

## Support & Contact

For questions or issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Review `output/run.log` for error details
3. Open an issue on GitHub (if repo is public)

***

**Built with:** Python 3.11, OpenAI GPT-5, HuggingFace Qwen, pdfplumber, pandas, Docker