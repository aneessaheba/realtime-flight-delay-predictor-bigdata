"""
Generate a code walkthrough PDF for the realtime-flight-delay-predictor project.
Run: python3 generate_walkthrough_pdf.py
Output: code_walkthrough.pdf
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Preformatted

OUTPUT = "code_walkthrough.pdf"

# ── Color palette ──────────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1a3a5c")
MID_BLUE    = colors.HexColor("#2563eb")
LIGHT_BLUE  = colors.HexColor("#dbeafe")
ACCENT      = colors.HexColor("#f59e0b")
CODE_BG     = colors.HexColor("#f1f5f9")
CODE_BORDER = colors.HexColor("#cbd5e1")
GREY        = colors.HexColor("#64748b")
LIGHT_GREY  = colors.HexColor("#f8fafc")
WHITE       = colors.white
BLACK       = colors.HexColor("#0f172a")


def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["title"] = ParagraphStyle(
        "title", parent=base["Normal"],
        fontSize=28, leading=34, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"],
        fontSize=13, leading=18, textColor=colors.HexColor("#bfdbfe"),
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4,
    )
    styles["meta"] = ParagraphStyle(
        "meta", parent=base["Normal"],
        fontSize=10, leading=14, textColor=colors.HexColor("#93c5fd"),
        fontName="Helvetica", alignment=TA_CENTER,
    )
    styles["section_num"] = ParagraphStyle(
        "section_num", parent=base["Normal"],
        fontSize=9, leading=12, textColor=MID_BLUE,
        fontName="Helvetica-Bold", spaceAfter=2,
    )
    styles["section_title"] = ParagraphStyle(
        "section_title", parent=base["Normal"],
        fontSize=16, leading=20, textColor=DARK_BLUE,
        fontName="Helvetica-Bold", spaceAfter=6,
    )
    styles["file_label"] = ParagraphStyle(
        "file_label", parent=base["Normal"],
        fontSize=11, leading=15, textColor=WHITE,
        fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=10, leading=15, textColor=BLACK,
        fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=base["Normal"],
        fontSize=10, leading=14, textColor=BLACK,
        fontName="Helvetica", leftIndent=16, spaceAfter=3,
        bulletIndent=4,
    )
    styles["sub_bullet"] = ParagraphStyle(
        "sub_bullet", parent=base["Normal"],
        fontSize=9.5, leading=13, textColor=GREY,
        fontName="Helvetica", leftIndent=32, spaceAfter=2,
        bulletIndent=20,
    )
    styles["label"] = ParagraphStyle(
        "label", parent=base["Normal"],
        fontSize=9, leading=12, textColor=GREY,
        fontName="Helvetica-Bold", spaceAfter=2,
    )
    styles["code"] = ParagraphStyle(
        "code", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"),
        fontName="Courier", leftIndent=8, rightIndent=8,
        spaceAfter=0, spaceBefore=0,
    )
    styles["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=GREY,
        fontName="Helvetica-Oblique", alignment=TA_CENTER,
    )
    styles["result_label"] = ParagraphStyle(
        "result_label", parent=base["Normal"],
        fontSize=9, leading=12, textColor=DARK_BLUE,
        fontName="Helvetica-Bold",
    )
    styles["result_val"] = ParagraphStyle(
        "result_val", parent=base["Normal"],
        fontSize=9, leading=12, textColor=BLACK,
        fontName="Helvetica",
    )
    styles["toc_title"] = ParagraphStyle(
        "toc_title", parent=base["Normal"],
        fontSize=13, leading=18, textColor=DARK_BLUE,
        fontName="Helvetica-Bold", spaceAfter=10,
    )
    styles["toc_item"] = ParagraphStyle(
        "toc_item", parent=base["Normal"],
        fontSize=10, leading=16, textColor=BLACK,
        fontName="Helvetica", leftIndent=12,
    )
    styles["note"] = ParagraphStyle(
        "note", parent=base["Normal"],
        fontSize=9.5, leading=13, textColor=colors.HexColor("#92400e"),
        fontName="Helvetica-Oblique", leftIndent=8,
    )
    return styles


def code_block(lines, styles, bg=CODE_BG):
    """Return a Table that renders as a shaded code block."""
    content = []
    for line in lines:
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        content.append(Paragraph(safe, styles["code"]))
    t = Table([[content]], colWidths=[6.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 0.5, CODE_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def file_header(filename, role_tag, styles, color=MID_BLUE):
    """Blue pill-style file header bar."""
    tag_cell = Paragraph(role_tag, ParagraphStyle(
        "tag", parent=styles["file_label"],
        fontSize=8, textColor=colors.HexColor("#bfdbfe"),
        fontName="Helvetica",
    ))
    name_cell = Paragraph(filename, styles["file_label"])
    t = Table([[name_cell, tag_cell]], colWidths=[4.5 * inch, 2.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (1, 0), (1, 0), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [color]),
    ]))
    return t


def section_divider(number, title, styles):
    return [
        HRFlowable(width="100%", thickness=2, color=MID_BLUE, spaceAfter=6),
        Paragraph(f"SECTION {number}", styles["section_num"]),
        Paragraph(title, styles["section_title"]),
        Spacer(1, 4),
    ]


def bullet(text, styles, sub=False):
    s = styles["sub_bullet"] if sub else styles["bullet"]
    prefix = "◦" if sub else "•"
    return Paragraph(f"{prefix}  {text}", s)


def kv_table(rows, styles, col_w=(2.2 * inch, 4.3 * inch)):
    """Two-column label/value table."""
    data = [
        [Paragraph(k, styles["result_label"]), Paragraph(v, styles["result_val"])]
        for k, v in rows
    ]
    t = Table(data, colWidths=list(col_w))
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, -1), LIGHT_GREY),
        ("BACKGROUND",   (1, 0), (1, -1), WHITE),
        ("BOX",          (0, 0), (-1, -1), 0.5, CODE_BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, CODE_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def metrics_table(headers, rows, styles):
    header_row = [Paragraph(h, ParagraphStyle(
        "th", parent=styles["body"], fontSize=9,
        fontName="Helvetica-Bold", textColor=WHITE,
    )) for h in headers]
    body_rows = [
        [Paragraph(str(c), styles["result_val"]) for c in row]
        for row in rows
    ]
    col_w = [6.5 * inch / len(headers)] * len(headers)
    t = Table([header_row] + body_rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), DARK_BLUE),
        ("BACKGROUND",   (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("BOX",          (0, 0), (-1, -1), 0.5, CODE_BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.3, CODE_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── Cover page ─────────────────────────────────────────────────────────────────

def cover_page(styles):
    story = []

    # Top banner
    banner_text = Paragraph(
        "Real-Time Flight Delay Predictor", styles["title"]
    )
    subtitle = Paragraph(
        "Complete Code Walkthrough — All Files Explained", styles["subtitle"]
    )
    meta = Paragraph(
        "SJSU DATA-228 · Spring 2026 · Anees Saheba Guddi", styles["meta"]
    )

    banner_data = [[banner_text], [subtitle], [Spacer(1, 6)], [meta]]
    banner = Table(banner_data, colWidths=[6.5 * inch])
    banner.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), DARK_BLUE),
        ("LEFTPADDING",  (0, 0), (-1, -1), 24),
        ("RIGHTPADDING", (0, 0), (-1, -1), 24),
        ("TOPPADDING",   (0, 0), (0, 0), 36),
        ("BOTTOMPADDING",(0, -1), (-1, -1), 36),
        ("TOPPADDING",   (0, 1), (-1, -1), 4),
    ]))
    story.append(banner)
    story.append(Spacer(1, 20))

    # Pipeline flow diagram (text-based)
    story.append(Paragraph("End-to-End Pipeline", ParagraphStyle(
        "flow_title", parent=styles["section_title"], fontSize=12, spaceAfter=8
    )))

    flow_lines = [
        "  BTS CSV Data  (raw airline on-time performance files)",
        "       ↓",
        "  [1] Ingestion  →  ingest_bts_to_hdfs.py  →  HDFS Parquet",
        "       ↓",
        "  [2] Training   →  train_local.py          →  LR + GBT PipelineModels",
        "       ↓",
        "  [3a] Batch     →  batch_inference.py      →  HDFS batch_predictions/",
        "       ↓                                          ↘",
        "  [3b] Producer  →  kafka_producer.py       →  Kafka topic: flight-events",
        "       ↓",
        "  [3c] Consumer  →  streaming_consumer.py   →  HDFS streaming_predictions/",
        "       ↓                                          ↗",
        "  [4] Benchmark  →  benchmark.py            →  benchmark_report.json",
    ]
    story.append(code_block(flow_lines, styles, bg=LIGHT_BLUE))
    story.append(Spacer(1, 20))

    # File index table
    story.append(Paragraph("Files Covered in This Document", ParagraphStyle(
        "fl", parent=styles["section_title"], fontSize=12, spaceAfter=8
    )))

    file_rows = [
        ["#", "File", "Owner", "Purpose"],
        ["1", "docker-compose.yml", "Rish", "7-service infrastructure"],
        ["2", "scripts/setup_hdfs.sh", "Kartheek", "Create HDFS directories"],
        ["3", "scripts/run_pipeline.sh", "Rish", "Orchestrate full pipeline (7 steps)"],
        ["4", "src/ingestion/ingest_bts_to_hdfs.py", "Kartheek", "CSV → HDFS Parquet"],
        ["5", "src/training/train_local.py", "Manjot", "LR + GBT training & CV"],
        ["6", "src/streaming/kafka_producer.py", "Keon", "Replay BTS data to Kafka"],
        ["7", "src/streaming/streaming_consumer.py", "Keon", "Real-time Spark inference"],
        ["8", "src/batch/batch_inference.py", "Kartheek", "Static batch scoring"],
        ["9", "src/evaluation/benchmark.py", "Anees", "Compare batch vs. streaming"],
        ["10", "models/benchmark_report.json", "Anees", "Final results output"],
    ]

    header = [Paragraph(c, ParagraphStyle("th2", parent=styles["body"], fontSize=9,
                fontName="Helvetica-Bold", textColor=WHITE)) for c in file_rows[0]]
    body = [[Paragraph(str(c), styles["result_val"]) for c in row] for row in file_rows[1:]]
    t = Table([header] + body, colWidths=[0.3*inch, 2.5*inch, 1.1*inch, 2.6*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), DARK_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("BOX",            (0, 0), (-1, -1), 0.5, CODE_BORDER),
        ("INNERGRID",      (0, 0), (-1, -1), 0.3, CODE_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)

    story.append(PageBreak())
    return story


# ── Section builders ───────────────────────────────────────────────────────────

def section_docker(styles):
    s = []
    s += section_divider("1", "docker-compose.yml — Infrastructure", styles)
    s.append(file_header("docker-compose.yml", "Infrastructure · Owner: Rish", styles, DARK_BLUE))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "Defines every service the system needs as Docker containers connected on a shared "
        "bridge network called <b>flight-net</b>. Running <code>docker compose up -d</code> "
        "starts the entire infrastructure in dependency order.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Services at a Glance", styles["label"]))
    s.append(kv_table([
        ("kafka",          "Apache Kafka 3.7 in KRaft mode (no Zookeeper). Ports 9092 (internal) and 9093 (external/host). Auto-creates topics, 24-hour log retention."),
        ("kafka-init",     "One-shot container that runs after Kafka is healthy and creates the 'flight-events' topic with 4 partitions, replication factor 1."),
        ("hdfs-namenode",  "Hadoop 3.2.1 namenode. Exposes web UI on port 9870 and RPC on port 9000. Stores filesystem metadata on a named Docker volume."),
        ("hdfs-datanode",  "Hadoop datanode. Stores actual block data. Starts only after namenode is healthy."),
        ("spark-master",   "Spark 3.5 standalone master. Web UI port 8080, cluster RPC port 7077. Mounts ./src and ./data as read-only volumes."),
        ("spark-worker-1/2","Two workers, each with 4 cores and 4 GB RAM, registered to spark-master:7077."),
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Key design choices:", styles["label"]))
    s.append(bullet("KRaft mode removes the Zookeeper dependency, simplifying the stack from 8 services to 7.", styles))
    s.append(bullet("All services have <b>healthchecks</b> so Docker waits for real readiness before starting dependents.", styles))
    s.append(bullet("HDFS volumes are named (not bind-mounts), so data survives container restarts.", styles))
    s.append(bullet("Kafka message max is 10 MB to handle large JSON flight batches.", styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Listener configuration:", styles["label"]))
    s.append(code_block([
        "PLAINTEXT://kafka:9092    ← Spark / consumer talks to Kafka inside Docker",
        "EXTERNAL://localhost:9093 ← kafka_producer.py on your laptop talks to Kafka",
    ], styles))
    s.append(PageBreak())
    return s


def section_setup_hdfs(styles):
    s = []
    s += section_divider("2", "scripts/setup_hdfs.sh — HDFS Directory Setup", styles)
    s.append(file_header("scripts/setup_hdfs.sh", "Script · Owner: Kartheek", styles))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "A short shell script that runs Hadoop CLI commands inside the namenode container "
        "to create the HDFS directory tree before any data is written.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Directories created:", styles["label"]))
    s.append(code_block([
        "hdfs://hdfs-namenode:9000/data/flights        ← raw Parquet ingestion target",
        "hdfs://hdfs-namenode:9000/models              ← trained ML pipeline artifacts",
        "hdfs://hdfs-namenode:9000/output/batch_predictions",
        "hdfs://hdfs-namenode:9000/output/streaming_predictions",
        "hdfs://hdfs-namenode:9000/checkpoints/streaming ← Spark checkpoint dir",
    ], styles))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Uses <code>hdfs dfs -mkdir -p</code> (idempotent — safe to rerun). "
        "Sets permissions to 777 so Spark workers can write without credential issues.",
        styles["body"]
    ))
    s.append(Spacer(1, 12))
    return s


def section_run_pipeline(styles):
    s = []
    s += section_divider("3", "scripts/run_pipeline.sh — Full Pipeline Orchestration", styles)
    s.append(file_header("scripts/run_pipeline.sh", "Script · Owner: Rish", styles))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "The master orchestration script. It runs all 7 pipeline steps in sequence, "
        "logs everything to <code>./logs/pipeline_TIMESTAMP.log</code>, and can skip "
        "any step via flags. All spark-submit calls are parameterised via environment "
        "variables with sensible defaults.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("7 steps in order:", styles["label"]))
    steps = [
        ("Step 1", "setup_hdfs()", "Runs setup_hdfs.sh to create HDFS directories."),
        ("Step 2", "ingest_data()", "spark-submit ingest_bts_to_hdfs.py for years 2018–2024."),
        ("Step 3", "train_models()", "spark-submit train_model.py with 5-fold CV on cluster."),
        ("Step 4", "start_streaming_consumer()", "spark-submit streaming_consumer.py in background; waits 30s for init."),
        ("Step 5", "run_kafka_producer()", "python3 kafka_producer.py streams 2024 data; waits 60s after; kills consumer."),
        ("Step 6", "run_batch_inference()", "spark-submit batch_inference.py on 2024 test data."),
        ("Step 7", "run_benchmark()", "spark-submit benchmark.py; writes benchmark_report.json."),
    ]
    data = [[Paragraph(a, styles["result_label"]), Paragraph(b, ParagraphStyle(
        "mono_sm", parent=styles["code"], fontSize=8)),
        Paragraph(c, styles["result_val"])] for a, b, c in steps]
    t = Table(data, colWidths=[0.75*inch, 2.5*inch, 3.25*inch])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_GREY]),
        ("BOX",           (0, 0), (-1, -1), 0.5, CODE_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, CODE_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    s.append(t)
    s.append(Spacer(1, 8))

    s.append(Paragraph("CLI flags:", styles["label"]))
    s.append(code_block([
        "bash scripts/run_pipeline.sh --skip-ingest --skip-training",
        "# Useful when models are already trained — jumps straight to streaming + benchmark",
    ], styles))
    s.append(Paragraph(
        "An EXIT trap calls cleanup() which kills the background consumer PID on any failure, "
        "preventing orphan Spark jobs.",
        styles["body"]
    ))
    s.append(PageBreak())
    return s


def section_ingestion(styles):
    s = []
    s += section_divider("4", "src/ingestion/ingest_bts_to_hdfs.py — Data Ingestion", styles)
    s.append(file_header("src/ingestion/ingest_bts_to_hdfs.py", "Data · Owner: Kartheek", styles))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "Reads raw BTS Airline On-Time Performance CSV files (2018–2024, ~35 GB compressed), "
        "cleans them, creates the binary label column, and writes Parquet to HDFS partitioned "
        "by YEAR and MONTH.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Step-by-step inside main():", styles["label"]))
    s.append(bullet("<b>resolve_csv_paths()</b> — supports both year-subdirectory layouts and flat directories.", styles))
    s.append(bullet("<b>read_raw_csv()</b> — reads all CSVs as strings initially (inferSchema=false avoids type mismatches). Renames mixed-case BTS columns to standard uppercase names via BTS_COLUMN_MAP.", styles))
    s.append(bullet("<b>clean_and_transform()</b> — drops rows where ARR_DELAY is null (cancelled/diverted flights), fills delay breakdown columns with 0 for on-time flights, drops rows missing critical features.", styles))
    s.append(bullet("<b>Label creation</b> — <code>label = 1 if ARR_DELAY > 15 else 0</code>. Binary classification.", styles))
    s.append(bullet("<b>write_to_hdfs()</b> — writes Parquet partitioned by YEAR and MONTH for efficient year-filtered reads downstream. Snappy compression.", styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Column rename map (sample):", styles["label"]))
    s.append(code_block([
        "BTS_COLUMN_MAP = {",
        '    "Reporting_Airline": "OP_UNIQUE_CARRIER",',
        '    "CRSDepTime":        "CRS_DEP_TIME",',
        '    "ArrDelay":          "ARR_DELAY",',
        '    "DayOfWeek":         "DAY_OF_WEEK",',
        "    ... (18 mappings total)",
        "}",
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Partitioning strategy:", styles["label"]))
    s.append(Paragraph(
        "Repartitions to ~500k rows per Parquet file using <code>repartition(n, 'YEAR', 'MONTH')</code>. "
        "When batch_inference.py later filters to <code>YEAR=2024</code>, Spark uses partition pruning "
        "to skip all other years — dramatically reducing I/O on 35 GB of data.",
        styles["body"]
    ))
    s.append(PageBreak())
    return s


def section_training(styles):
    s = []
    s += section_divider("5", "src/training/train_local.py — ML Model Training", styles)
    s.append(file_header("src/training/train_local.py", "ML · Owner: Manjot", styles, colors.HexColor("#7c3aed")))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "The core ML script. Trains two Spark MLlib models — Logistic Regression (baseline) "
        "and Gradient Boosted Trees (primary) — inside full Spark ML Pipelines with "
        "cross-validation. Runs in local[*] mode so no cluster is needed for development.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Features used:", styles["label"]))
    s.append(kv_table([
        ("Categorical (3)",  "OP_UNIQUE_CARRIER, ORIGIN, DEST — airline carrier code, origin airport, destination airport."),
        ("Numeric (6)",      "DAY_OF_WEEK, CRS_DEP_TIME, DEP_DELAY, CRS_ELAPSED_TIME, DISTANCE, MONTH."),
        ("Label",           "ARR_DELAY > 15 min → 1 (delayed), else 0 (on-time)."),
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Shared preprocessing pipeline (4 stages):", styles["label"]))
    s.append(code_block([
        "Stage 1: StringIndexer  (×3)  — encodes carrier/origin/dest to numeric index",
        "Stage 2: Imputer              — fills missing numeric values with median",
        "Stage 3: VectorAssembler      — combines all features → raw_features vector",
        "Stage 4: StandardScaler       — normalizes (withMean=True, withStd=True)",
        "                                (critical for LR; harmless for GBT)",
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Class weighting:", styles["label"]))
    s.append(Paragraph(
        "About 77% of flights are on-time, only 23% are delayed — an imbalanced dataset. "
        "The script computes <code>weight = total / (n_classes × class_count)</code> for each class "
        "and assigns it via a <code>class_weight</code> column, so the model penalises missed "
        "delays more heavily than incorrect on-time predictions.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Cross-validation grid:", styles["label"]))
    s.append(metrics_table(
        ["Model", "Hyperparameter", "Values Searched", "Combos"],
        [
            ["GBT",  "maxDepth / maxIter / stepSize", "{4,5} × {30,50} × {0.05,0.1}", "8 × 3 folds = 24 jobs"],
            ["LR",   "regParam / elasticNetParam",    "{0.001,0.01,0.1} × {0.0,0.5}", "6 × 3 folds = 18 jobs"],
        ],
        styles
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Model serialization:", styles["label"]))
    s.append(Paragraph(
        "Both models are saved as complete <b>PipelineModel</b> objects "
        "(<code>model.write().overwrite().save(path)</code>). This means the fitted "
        "StringIndexer vocabularies, imputer medians, scaler statistics, AND model weights "
        "are all bundled together. When the streaming consumer loads the pipeline later, "
        "it applies the exact same transformations — preventing training/serving feature skew.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Achieved results (Dec 2021 BTS data):", styles["label"]))
    s.append(metrics_table(
        ["Model", "AUC-ROC", "AUC-PR", "F1", "Precision", "Recall"],
        [
            ["GBTClassifier",      "0.9341", "—",     "0.9018", "0.9022", "0.9015"],
            ["LogisticRegression", "~0.85",  "—",     "~0.82",  "~0.83",  "~0.82"],
        ],
        styles
    ))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "GBT F1=0.9018 far exceeds the mid-term target of ≥ 0.70.",
        ParagraphStyle("note_green", parent=styles["note"], textColor=colors.HexColor("#065f46"))
    ))
    s.append(PageBreak())
    return s


def section_producer(styles):
    s = []
    s += section_divider("6", "src/streaming/kafka_producer.py — Kafka Producer", styles)
    s.append(file_header("src/streaming/kafka_producer.py", "Streaming · Owner: Keon", styles, colors.HexColor("#b45309")))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "Replays 2024 BTS CSV records as a live JSON stream to Kafka, simulating real-time "
        "flight event ingestion. Rate-limited to a configurable messages/second target.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Key functions:", styles["label"]))
    s.append(bullet("<b>build_producer()</b> — connects with acks='all' (durable), gzip compression, 64 KB batches, 10 ms linger for throughput. Retries 5× if Kafka is not yet ready.", styles))
    s.append(bullet("<b>discover_csv_files()</b> — walks the input directory recursively for all .csv files, sorted by name to ensure deterministic replay order.", styles))
    s.append(bullet("<b>row_to_dict()</b> — cleans each CSV row, casts numeric columns to float (None if empty/NA), and adds producer_ts = time.time(). This timestamp is the start of the latency clock.", styles))
    s.append(bullet("<b>produce_records()</b> — token-bucket rate limiter: computes interval = 1/rate seconds, sleeps for the remaining time in each cycle after send. Uses ORIGIN as the Kafka partition key.", styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Partition key rationale:", styles["label"]))
    s.append(Paragraph(
        "Using the departure airport (ORIGIN) as the Kafka partition key means all flights from "
        "the same origin are always routed to the same partition. This preserves temporal ordering "
        "per origin, which matters for any analysis that groups flights by departure airport.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Rate limiting:", styles["label"]))
    s.append(code_block([
        "interval = 1.0 / rate  # e.g. 100 msg/s → interval = 0.01s",
        "for record in records:",
        "    send_start = time.time()",
        "    producer.send(topic, key=origin, value=record)",
        "    elapsed = time.time() - send_start",
        "    if interval - elapsed > 0:",
        "        time.sleep(interval - elapsed)  # sleep only remaining time",
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("CLI usage:", styles["label"]))
    s.append(code_block([
        "python src/streaming/kafka_producer.py \\",
        "    --input-path data/raw/2024  \\",
        "    --kafka-bootstrap localhost:9093 \\",
        "    --topic flight-events  \\",
        "    --rate 500             \\",
        "    --loop                 # replay indefinitely for long-running tests",
    ], styles))
    s.append(PageBreak())
    return s


def section_consumer(styles):
    s = []
    s += section_divider("7", "src/streaming/streaming_consumer.py — Spark Streaming Consumer", styles)
    s.append(file_header("src/streaming/streaming_consumer.py", "Streaming · Owner: Keon", styles, colors.HexColor("#b45309")))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "A Spark Structured Streaming application that subscribes to the Kafka topic, "
        "deserializes JSON flight events, runs them through the GBT PipelineModel, "
        "measures end-to-end latency, and writes predictions to HDFS every 10 seconds.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Data flow inside the job:", styles["label"]))
    s.append(code_block([
        "Kafka topic 'flight-events'",
        "  → read_kafka_stream()      # subscribe, max 10k offsets per trigger",
        "  → parse_kafka_messages()   # from_json() with explicit FLIGHT_EVENT_SCHEMA",
        "  → foreachBatch handler every 10 seconds:",
        "       model.transform(batch_df)          # GBT pipeline inference",
        "       latency_ms = (consumer_ts - producer_ts) * 1000",
        "       output_df.write.parquet(output_path, mode='append')",
        "       log latency stats (mean, p50, p95, p99)",
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Why explicit schema (not inferSchema)?", styles["label"]))
    s.append(Paragraph(
        "Spark's JSON schema inference requires reading a sample of the data first, "
        "adding latency to stream startup. An explicit <b>StructType</b> schema makes parsing "
        "deterministic and faster. It also fails clearly if the producer sends an unexpected field "
        "instead of silently dropping it.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Back-pressure:", styles["label"]))
    s.append(Paragraph(
        "<code>maxOffsetsPerTrigger=10_000</code> caps Kafka reads per micro-batch. If the "
        "producer bursts to 50,000 messages, Spark processes them across 5 batches rather than "
        "all at once, keeping per-batch memory bounded.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Latency formula:", styles["label"]))
    s.append(code_block([
        "# producer_ts: epoch seconds stamped by kafka_producer.py at send time",
        "# consumer_ts: epoch seconds at the start of this micro-batch",
        "latency_ms = (consumer_ts - producer_ts) * 1000",
        "",
        "# Per the benchmark report, observed latency on Dec 2021 data:",
        "#   mean = 5,303 ms  |  p50 = 5,212 ms  |  p99 = 10,438 ms",
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Output columns written to HDFS:", styles["label"]))
    s.append(code_block([
        "batch_id, kafka_timestamp, kafka_partition, kafka_offset,",
        "YEAR, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, OP_UNIQUE_CARRIER,",
        "ORIGIN, DEST, CRS_DEP_TIME, DEP_DELAY, ARR_DELAY, DISTANCE,",
        "prediction, probability, rawPrediction,",
        "producer_ts, consumer_ts, latency_ms",
    ], styles))
    s.append(PageBreak())
    return s


def section_batch(styles):
    s = []
    s += section_divider("8", "src/batch/batch_inference.py — Batch Inference", styles)
    s.append(file_header("src/batch/batch_inference.py", "Batch · Owner: Kartheek", styles, colors.HexColor("#065f46")))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "The <b>control group</b> in the benchmark. Loads the 2024 test year from HDFS, "
        "applies the same saved GBT PipelineModel, evaluates classification metrics, "
        "and writes predictions to HDFS. No Kafka involved — pure batch Spark job.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Step-by-step:", styles["label"]))
    s.append(bullet("<b>load_test_data()</b> — reads HDFS Parquet, filters to YEAR=2024 using partition pruning (fast). Casts numeric columns to DoubleType for the pipeline.", styles))
    s.append(bullet("<b>load_model()</b> — PipelineModel.load() from HDFS. This is the same artifact training saved.", styles))
    s.append(bullet("<b>run_inference()</b> — model.transform() followed by .cache() and .count() to force Spark to materialize the lazy computation and time it accurately.", styles))
    s.append(bullet("<b>evaluate_predictions()</b> — computes AUC-ROC, AUC-PR, F1, precision, recall, accuracy, AND a full confusion matrix (TP/TN/FP/FN).", styles))
    s.append(bullet("<b>write_predictions()</b> — appends output Parquet to HDFS. The benchmark script later reads this.", styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Confusion matrix computation:", styles["label"]))
    s.append(code_block([
        "tp = df.filter((prediction==1) & (label==1)).count()",
        "tn = df.filter((prediction==0) & (label==0)).count()",
        "fp = df.filter((prediction==1) & (label==0)).count()  # false alarm",
        "fn = df.filter((prediction==0) & (label==1)).count()  # missed delay",
    ], styles))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Batch results (537,183 records from Dec 2021 BTS data):", styles["label"]))
    s.append(metrics_table(
        ["Metric", "Batch Value"],
        [
            ["AUC-ROC",   "0.9386"],
            ["AUC-PR",    "0.8877"],
            ["F1",        "0.9029"],
            ["Precision", "0.9058"],
            ["Recall",    "0.9011"],
            ["Accuracy",  "0.9011"],
            ["% Delayed", "23.7%"],
        ],
        styles
    ))
    s.append(PageBreak())
    return s


def section_benchmark(styles):
    s = []
    s += section_divider("9", "src/evaluation/benchmark.py — Benchmark & Evaluation", styles)
    s.append(file_header("src/evaluation/benchmark.py", "Evaluation · Owner: Anees", styles, colors.HexColor("#991b1b")))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "The evaluation module — your code. Loads both batch and streaming prediction outputs, "
        "computes classification metrics, latency and throughput statistics, applies "
        "Differential Privacy noise for safe public reporting, and runs an LSH-based "
        "anomaly detection pass. Writes everything to benchmark_report.json.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    # DP section
    s.append(Paragraph("Technique 1 — Differential Privacy (Laplace Mechanism)", ParagraphStyle(
        "sub_head", parent=styles["label"], fontSize=10, textColor=DARK_BLUE,
        fontName="Helvetica-Bold", spaceAfter=4,
    )))
    s.append(Paragraph(
        "After computing true aggregate metrics (F1, AUC-ROC, accuracy), the script adds "
        "Laplace noise before writing the public-facing report. This ensures that no individual "
        "flight record's true delay label can be inferred from published statistics.",
        styles["body"]
    ))
    s.append(code_block([
        "# Laplace mechanism: noise ~ Laplace(0, sensitivity / epsilon)",
        "# sensitivity = 0.01  (one record changes accuracy by at most 1/n)",
        "# epsilon     = 1.0   (moderate privacy budget)",
        "# noise scale = 0.01 / 1.0 = 0.01",
        "",
        "def dp_metric(value, epsilon=1.0, sensitivity=0.01):",
        "    mech = diffprivlib.Laplace(epsilon=epsilon, sensitivity=sensitivity)",
        "    return clip(mech.randomise(value), 0.0, 1.0)",
        "",
        "# Example: true F1=0.9029 → published as F1≈0.911 (noisy but close)",
    ], styles))
    s.append(Spacer(1, 8))

    # LSH section
    s.append(Paragraph("Technique 2 — LSH Anomaly Detection (datasketch MinHashLSH)", ParagraphStyle(
        "sub_head2", parent=styles["label"], fontSize=10, textColor=DARK_BLUE,
        fontName="Helvetica-Bold", spaceAfter=4,
    )))
    s.append(Paragraph(
        "After scoring, the script builds MinHash signatures for 5,000 streaming predictions "
        "using categorical features (carrier + origin + dest). It inserts all signatures into "
        "a MinHashLSH index and queries for approximate nearest neighbors (Jaccard ≥ 0.5). "
        "Any pair of similar flights that receive different predicted labels is flagged as anomalous.",
        styles["body"]
    ))
    s.append(code_block([
        "lsh = MinHashLSH(threshold=0.5, num_perm=128)",
        "for i, row in enumerate(sample):",
        "    m = MinHash(num_perm=128)",
        "    for token in [carrier, origin, dest]:",
        "        m.update(token.encode())",
        "    lsh.insert(f'flight_{i}', m)",
        "",
        "# Query each flight for similar neighbors",
        "# Flag if neighbor predicts a DIFFERENT class",
        "anomaly_rate = anomalous_pairs / total_pairs_checked",
    ], styles))
    s.append(Spacer(1, 8))

    # Latency & throughput
    s.append(Paragraph("compute_latency_stats() and compute_throughput_stats():", styles["label"]))
    s.append(Paragraph(
        "Uses Spark aggregate functions (<code>percentile_approx</code>, <code>stddev</code>, <code>mean</code>) "
        "to compute latency statistics across all streaming prediction records. Throughput is "
        "computed per batch_id using min/max consumer_ts as the batch window.",
        styles["body"]
    ))
    s.append(Spacer(1, 6))

    s.append(Paragraph("Output structure of benchmark_report.json:", styles["label"]))
    s.append(code_block([
        "{",
        '  "batch_metrics":              { AUC-ROC, F1, precision, recall, accuracy },',
        '  "streaming_metrics":          { record_count, prediction_distribution },',
        '  "batch_metrics_dp_reported":  { same fields + dp_epsilon, dp_mechanism },',
        '  "streaming_metrics_dp_reported": { ... },',
        '  "streaming_latency":          { mean_ms, p50_ms, p90_ms, p95_ms, p99_ms },',
        '  "streaming_throughput":       { mean/max events_per_sec },',
        '  "lsh_anomaly_detection":      { pairs_checked, anomaly_rate, interpretation }',
        "}",
    ], styles))
    s.append(PageBreak())
    return s


def section_results(styles):
    s = []
    s += section_divider("10", "models/benchmark_report.json — Final Results", styles)
    s.append(file_header("models/benchmark_report.json", "Output · Owner: Anees", styles, colors.HexColor("#991b1b")))
    s.append(Spacer(1, 10))

    s.append(Paragraph(
        "The final output of the entire pipeline — produced by benchmark.py and used as "
        "the source of truth for the course report and paper.",
        styles["body"]
    ))
    s.append(Spacer(1, 8))

    s.append(Paragraph("Batch Inference Results:", styles["label"]))
    s.append(metrics_table(
        ["Metric", "Value"],
        [
            ["Records scored",    "537,183"],
            ["AUC-ROC",           "0.9386"],
            ["AUC-PR",            "0.8877"],
            ["F1",                "0.9029"],
            ["Precision",         "0.9058"],
            ["Recall",            "0.9011"],
            ["Accuracy",          "0.9011"],
            ["% Predicted Delayed", "23.7%"],
        ],
        styles
    ))
    s.append(Spacer(1, 10))

    s.append(Paragraph("Streaming Latency (62,815 records):", styles["label"]))
    s.append(metrics_table(
        ["Metric", "Value"],
        [
            ["Mean latency",   "5,303 ms"],
            ["p50 latency",    "5,212 ms"],
            ["p90 latency",    "9,394 ms"],
            ["p95 latency",    "9,921 ms"],
            ["p99 latency",    "10,438 ms"],
            ["Min latency",    "144 ms"],
            ["Max latency",    "10,613 ms"],
        ],
        styles
    ))
    s.append(Spacer(1, 10))

    s.append(Paragraph("LSH Anomaly Detection:", styles["label"]))
    s.append(metrics_table(
        ["Metric", "Value"],
        [
            ["Sample size",      "5,000 predictions"],
            ["Pairs checked",    "~24.99 million"],
            ["Anomalous pairs",  "0"],
            ["Anomaly rate",     "0.0%"],
            ["Interpretation",   "Predictions are consistent across similar flights"],
        ],
        styles
    ))
    s.append(Spacer(1, 10))

    s.append(Paragraph("Known Issue — Streaming 0% delayed predictions:", styles["label"]))
    s.append(Paragraph(
        "The streaming consumer predicted 0% delayed for all 62,815 streaming records. "
        "The most likely cause: ARR_DELAY is a post-flight column that is null in real-time "
        "events (the flight hasn't landed yet). The imputer fills it with the training median "
        "(a non-delay value), which shifts the decision boundary, causing the model to predict "
        "class 0 for every record. Fix: remove ARR_DELAY from features for real-time inference "
        "and rely only on pre-departure features (DEP_DELAY, CRS_DEP_TIME, DISTANCE, etc.).",
        styles["body"]
    ))
    s.append(Spacer(1, 12))

    s.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))
    s.append(Paragraph(
        "Generated by Claude Code · SJSU DATA-228 · April 2026",
        styles["caption"]
    ))
    return s


# ── Build document ─────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Real-Time Flight Delay Predictor — Code Walkthrough",
        author="Anees Saheba Guddi · SJSU DATA-228",
    )

    styles = build_styles()
    story = []

    story += cover_page(styles)
    story += section_docker(styles)
    story += section_setup_hdfs(styles)
    story += section_run_pipeline(styles)
    story += section_ingestion(styles)
    story += section_training(styles)
    story += section_producer(styles)
    story += section_consumer(styles)
    story += section_batch(styles)
    story += section_benchmark(styles)
    story += section_results(styles)

    doc.build(story)
    print(f"PDF written to: {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
