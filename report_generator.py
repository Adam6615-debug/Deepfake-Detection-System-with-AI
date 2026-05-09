"""
report_generator.py
-------------------
HTML/PDF report generator for deepfake detection results.
Creates professional reports with scores, verdicts, and visualizations.
"""

import os
import json
import base64
from datetime import datetime
from jinja2 import Template

# Try to import fpdf2 (pure Python, no system deps — pip install fpdf2)
FPDF_AVAILABLE = False
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    print("[WARNING] fpdf2 not available - run: pip install fpdf2")


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for embedding in HTML."""
    if not image_path or not os.path.exists(image_path):
        return None
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def generate_html_report(report: dict) -> str:
    """
    Generate HTML report from completed analysis report.
    
    Args:
        report: Completed report dictionary with all scores and analysis
        
    Returns:
        str: Path to generated HTML file
    """
    # Create HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detection Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007bff;
            margin: 0;
            font-size: 2.5em;
        }
        .header .timestamp {
            color: #666;
            font-size: 0.9em;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .section h2 {
            color: #007bff;
            margin-top: 0;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .image-preview {
            text-align: center;
            margin: 20px 0;
        }
        .image-preview img {
            max-width: 400px;
            max-height: 300px;
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .scores-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .score-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }
        .score-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .score-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .verdict {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 1.2em;
            font-weight: bold;
        }
        .verdict.manipulated {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .verdict.authentic {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .verdict.suspicious {
            background-color: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }
        .evidence-list {
            list-style: none;
            padding: 0;
        }
        .evidence-list li {
            background: #f8f9fa;
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        .heatmap-section {
            margin: 30px 0;
        }
        .heatmap-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .heatmap-images img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .no-heatmap {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .metadata {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-size: 0.9em;
            color: #666;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 10px;
        }
        .metadata-label {
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Deepfake Detection Report</h1>
            <div class="timestamp">Generated on {{ timestamp }}</div>
        </div>

        <div class="section">
            <h2>📷 Analyzed Image</h2>
            <div class="image-preview">
                {% if image_base64 %}
                <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Analyzed Image">
                {% else %}
                <p>Original image: {{ report.image_path }}</p>
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2>📊 Detection Scores</h2>
            <div class="scores-grid">
                <div class="score-card">
                    <div class="score-value">{{ "%.3f"|format(report.deepfake_score or 0) }}</div>
                    <div class="score-label">Deepfake Score</div>
                </div>
                <div class="score-card">
                    <div class="score-value">{{ "%.3f"|format(report.splicing_score or 0) }}</div>
                    <div class="score-label">Splicing Score</div>
                </div>
                <div class="score-card">
                    <div class="score-value">{{ "%.3f"|format(report.ai_generated_score or 0) }}</div>
                    <div class="score-label">AI-Generated Score</div>
                </div>
                <div class="score-card">
                    <div class="score-value">{{ "%.3f"|format(report.compression_score or 0) }}</div>
                    <div class="score-label">Compression Score</div>
                </div>
                {% if report.reverse_image_match %}
                <div class="score-card">
                    <div class="score-value">{{ "%.3f"|format(report.reverse_image_match.similarity or 0) }}</div>
                    <div class="score-label">Reverse Image Match</div>
                </div>
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2>⚖️ Final Verdict</h2>
            <div class="verdict {{ report.final_verdict.lower() }}">
                {% if report.final_verdict == 'MANIPULATED' %}
                    🚨 <strong>MANIPULATED</strong> - This image appears to be a deepfake or has been altered
                {% elif report.final_verdict == 'AUTHENTIC' %}
                    ✅ <strong>AUTHENTIC</strong> - This image appears to be genuine
                {% elif report.final_verdict == 'SUSPICIOUS' %}
                    ⚠️ <strong>SUSPICIOUS</strong> - This image shows signs of manipulation
                {% else %}
                    ❓ <strong>{{ report.final_verdict or 'UNKNOWN' }}</strong> - Analysis incomplete
                {% endif %}
            </div>
            <div class="metadata">
                <div class="metadata-grid">
                    <div class="metadata-label">Confidence:</div>
                    <div>{{ "%.1f"|format(report.final_confidence or 0) }}%</div>
                    <div class="metadata-label">Method:</div>
                    <div>{{ report.verdict_method or 'Weighted combination of all detectors' }}</div>
                </div>
            </div>
        </div>

        {% if report.evidence_flags %}
        <div class="section">
            <h2>🔍 Evidence Flags</h2>
            <ul class="evidence-list">
                {% for flag in report.evidence_flags %}
                <li>📌 {{ flag }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if report.shap_heatmap_path %}
        <div class="heatmap-section">
            <h2>🔥 Explainability Analysis</h2>
            <p>The heatmap below shows which regions of the image contributed most to the fake detection:</p>
            <div class="heatmap-images">
                <div>
                    <h4>Original Image</h4>
                    {% if image_base64 %}
                    <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Original">
                    {% endif %}
                </div>
                <div>
                    <h4>SHAP Heatmap (Red = Fake Indicators)</h4>
                    {% if shap_base64 %}
                    <img src="data:image/png;base64,{{ shap_base64 }}" alt="SHAP Heatmap">
                    {% endif %}
                </div>
            </div>
        </div>
        {% else %}
        <div class="heatmap-section">
            <h2>🔥 Explainability Analysis</h2>
            <div class="no-heatmap">
                SHAP heatmap not available - requires trained model for analysis
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>📋 Analysis Metadata</h2>
            <div class="metadata">
                <div class="metadata-grid">
                    <div class="metadata-label">Image Path:</div>
                    <div>{{ report.image_path }}</div>
                    <div class="metadata-label">Analysis Date:</div>
                    <div>{{ report.analysis_date or timestamp }}</div>
                    <div class="metadata-label">Deepfake Label:</div>
                    <div>{{ report.deepfake_label or 'N/A' }}</div>
                    <div class="metadata-label">Deepfake Confidence:</div>
                    <div>{{ "%.1f"|format(report.deepfake_confidence or 0) }}%</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    """

    # Prepare template data
    template_data = {
        'report': report,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_base64': encode_image_to_base64(report.get('image_path')),
        'shap_base64': encode_image_to_base64(report.get('shap_heatmap_path'))
    }

    # Render HTML
    template = Template(html_template)
    html_content = template.render(**template_data)

    # Save HTML file
    os.makedirs("outputs/reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"outputs/reports/deepfake_report_{timestamp}.html"
    
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[REPORT] HTML report saved to: {html_filename}")
    return html_filename


def generate_pdf_report(report: dict, html_path: str) -> str:
    """
    Generate a PDF report directly using fpdf2 (pure Python, no system libs).
    Falls back to returning the HTML path if fpdf2 is not installed.

    Args:
        report:    Completed report dictionary
        html_path: Path to the already-generated HTML file (used for fallback)

    Returns:
        str: Path to generated PDF file, or html_path if PDF unavailable
    """
    if not FPDF_AVAILABLE:
        print("[WARNING] PDF generation not available — run: pip install fpdf2")
        return html_path

    try:
        pdf_path = html_path.replace('.html', '.pdf')

        # ── Helpers ───────────────────────────────────────────────────────
        def verdict_color(verdict: str):
            v = (verdict or "").upper()
            if v == "MANIPULATED": return (220, 53, 69)    # red
            if v == "AUTHENTIC":   return (40, 167, 69)    # green
            if v == "SUSPICIOUS":  return (255, 193, 7)    # amber
            return (108, 117, 125)                          # grey

        def score_bar_color(score: float):
            if score >= 0.7: return (220, 53, 69)
            if score >= 0.4: return (255, 193, 7)
            return (40, 167, 69)

        def draw_score_bar(pdf, label, score, y):
            """Draw a labelled score bar at position y."""
            score = score or 0.0
            bar_x, bar_w, bar_h = 30, 150, 6
            pdf.set_xy(30, y)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(60, 8, safe(label), ln=0)
            # background
            pdf.set_fill_color(220, 220, 220)
            pdf.rect(90, y + 1, bar_w, bar_h, style="F")
            # filled portion
            r, g, b = score_bar_color(score)
            pdf.set_fill_color(r, g, b)
            pdf.rect(90, y + 1, bar_w * score, bar_h, style="F")
            # numeric value
            pdf.set_xy(245, y)
            pdf.set_font("Helvetica", "B", size=9)
            pdf.cell(20, 8, f"{score:.3f}", align="R")

        # ── Build PDF ─────────────────────────────────────────────────────
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        def safe(text: str) -> str:
            """Strip characters outside latin-1 so Helvetica never errors."""
            return (text or "").encode("latin-1", errors="replace").decode("latin-1")

        # ── Header ────────────────────────────────────────────────────────
        pdf.set_fill_color(0, 123, 255)
        pdf.rect(0, 0, 210, 28, style="F")
        pdf.set_font("Helvetica", "B", size=18)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(0, 7)
        pdf.cell(210, 10, safe("Deepfake Detection Report"), align="C")
        pdf.set_font("Helvetica", size=9)
        pdf.set_xy(0, 18)
        pdf.cell(210, 6,
                 safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                 align="C")
        pdf.ln(18)

        # ── Analysed image info ───────────────────────────────────────────
        pdf.set_font("Helvetica", "B", size=11)
        pdf.set_text_color(0, 123, 255)
        pdf.cell(0, 8, safe("Analysed Image"), ln=True)
        pdf.set_draw_color(0, 123, 255)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, safe(f"File: {report.get('filename', 'N/A')}"), ln=True)
        pdf.cell(0, 6, safe(f"Path: {report.get('image_path', 'N/A')}"), ln=True)
        pdf.cell(0, 6,
                 safe(f"Analysed at: {report.get('analyzed_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}"),
                 ln=True)
        pdf.ln(4)

        # Embed original image if it exists
        img_path = report.get("image_path", "")
        if img_path and os.path.exists(img_path):
            try:
                pdf.image(img_path, x=75, w=60)
                pdf.ln(4)
            except Exception:
                pass

        # ── Verdict box ───────────────────────────────────────────────────
        verdict = (report.get("final_verdict") or "UNKNOWN").upper()
        r, g, b = verdict_color(verdict)
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", size=14)
        pdf.cell(0, 14, safe(f"  Verdict: {verdict}"), ln=True, fill=True)
        pdf.set_text_color(60, 60, 60)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6,
                 safe(f"  Confidence: {report.get('final_confidence', 0):.1f}%   "
                      f"Risk Level: {report.get('risk_level', 'N/A')}"),
                 ln=True)
        pdf.ln(5)

        # ── Detection scores ──────────────────────────────────────────────
        pdf.set_font("Helvetica", "B", size=11)
        pdf.set_text_color(0, 123, 255)
        pdf.cell(0, 8, safe("Detection Scores"), ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        score_rows = [
            ("Deepfake Score (CNN)",    report.get("deepfake_score")),
            ("Splicing Score",          report.get("splicing_score")),
            ("AI-Generated Score",      report.get("ai_generated_score")),
            ("Compression Score",       report.get("compression_score")),
        ]
        for label, score in score_rows:
            draw_score_bar(pdf, label, score or 0.0, pdf.get_y())
            pdf.ln(10)

        # Reverse image match
        rim = report.get("reverse_image_match") or {}
        rim_sim = rim.get("similarity", 0.0)
        draw_score_bar(pdf, "Reverse Image Match", rim_sim, pdf.get_y())
        pdf.ln(10)

        # CNN label row
        pdf.set_xy(30, pdf.get_y())
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(60, 60, 60)
        cnn_label = report.get("deepfake_label", "N/A")
        cnn_conf  = report.get("deepfake_confidence", 0)
        pdf.cell(0, 6,
                 safe(f"CNN Prediction: {cnn_label}  ({cnn_conf:.1f}% confidence)"),
                 ln=True)
        pdf.ln(4)

        # ── Evidence flags ────────────────────────────────────────────────
        flags = report.get("evidence_flags") or []
        if flags:
            pdf.set_font("Helvetica", "B", size=11)
            pdf.set_text_color(0, 123, 255)
            pdf.cell(0, 8, safe("Evidence Flags"), ln=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(60, 60, 60)
            for flag in flags:
                pdf.set_x(14)
                pdf.cell(4, 6, "-", ln=0)
                pdf.multi_cell(0, 6, safe(flag))
            pdf.ln(4)

        # ── Reverse image search summary ──────────────────────────────────
        if rim:
            pdf.set_font("Helvetica", "B", size=11)
            pdf.set_text_color(0, 123, 255)
            pdf.cell(0, 8, safe("Reverse Image Search"), ln=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 6, safe(f"Verdict: {rim.get('verdict', 'N/A')}"), ln=True)
            pdf.cell(0, 6, safe(f"Similarity: {rim.get('similarity', 0):.4f}"), ln=True)
            pdf.cell(0, 6, safe(f"Matches found: {rim.get('matches_found', 0)}"), ln=True)
            pdf.cell(0, 6, safe(f"Database size: {rim.get('database_size', 0)}"), ln=True)
            bm = rim.get("best_match")
            if bm:
                pdf.cell(0, 6,
                         safe(f"Best match: {bm.get('image_path', 'N/A')} "
                              f"(label: {bm.get('label', 'N/A')}, "
                              f"similarity: {bm.get('avg_similarity', 0):.4f})"),
                         ln=True)
            pdf.ln(4)

        # ── Heatmap ───────────────────────────────────────────────────────
        heatmap_path = report.get("shap_heatmap_path")
        if heatmap_path and os.path.exists(heatmap_path):
            pdf.set_font("Helvetica", "B", size=11)
            pdf.set_text_color(0, 123, 255)
            pdf.cell(0, 8, safe("Explainability Heatmap"), ln=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 6,
                     safe(f"Method: {report.get('explainability_method', 'Gradient Analysis')}"),
                     ln=True)
            pdf.ln(2)
            try:
                pdf.image(heatmap_path, x=10, w=190)
            except Exception as e:
                pdf.cell(0, 6, safe(f"[Heatmap could not be embedded: {e}]"), ln=True)
            pdf.ln(4)

        # ── Footer ────────────────────────────────────────────────────────
        pdf.set_y(-20)
        pdf.set_font("Helvetica", "I", size=8)
        pdf.set_text_color(150, 150, 150)
        # plain hyphen instead of em dash — avoids latin-1 encoding error
        pdf.cell(0, 6,
                 safe("Generated by DeepfakeDetector - CET251 El Sewedy University"),
                 align="C")

        pdf.output(pdf_path)
        print(f"[REPORT] PDF report saved to: {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {str(e)}")
        return html_path


def create_final_report(report: dict, format: str = "both") -> dict:
    """
    Generate final report in specified format(s).
    
    Args:
        report: Completed analysis report
        format: "html", "pdf", or "both"
        
    Returns:
        dict: Updated report with file paths
    """
    print("[REPORT] Generating final report...")
    
    # Generate HTML report
    html_path = generate_html_report(report)
    report['html_report_path'] = html_path
    
    # Generate PDF if requested
    if format in ["pdf", "both"]:
        pdf_path = generate_pdf_report(report, html_path)
        report['pdf_report_path'] = pdf_path
    
    print("[REPORT] Report generation complete")
    return report
