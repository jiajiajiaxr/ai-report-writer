from __future__ import annotations

from pathlib import Path
import re
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import markdown

from .models import ChunkRecord, OutlineNode, SectionDraft, ValidationReport
from .utils import collect_citation_ids, dump_json, markdown_paragraphs, slugify, truncate


def compose_markdown(title: str, section_texts: list[str]) -> str:
    body = [f"# {title}", ""]
    for text in section_texts:
        body.append(text.strip())
        body.append("")
    return "\n".join(body).strip() + "\n"


def compose_structured_markdown(title: str, outline: list[OutlineNode], drafts: list[SectionDraft]) -> str:
    draft_map = {draft.node_id: draft.content.strip() for draft in drafts}
    body = [f"# {title}", ""]

    def render(node: OutlineNode, depth: int) -> None:
        if node.children:
            body.append(f"{'#' * min(depth + 1, 6)} {node.title}")
            body.append("")
            for child in node.children:
                render(child, depth + 1)
            return
        text = draft_map.get(node.node_id)
        if text:
            body.append(text)
            body.append("")

    for root in outline:
        render(root, 1)
    return "\n".join(body).strip() + "\n"


def export_markdown(report_dir: Path, title: str, markdown_text: str) -> Path:
    path = report_dir / f"{slugify(title)}.md"
    path.write_text(markdown_text, encoding="utf-8")
    return path


def export_html(report_dir: Path, title: str, markdown_text: str) -> Path:
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ max-width: 900px; margin: 40px auto; font-family: "Microsoft YaHei", "PingFang SC", sans-serif; line-height: 1.8; color: #1f2937; }}
    h1, h2, h3, h4 {{ color: #0f172a; }}
    code {{ background: #f3f4f6; padding: 0 4px; }}
    blockquote {{ border-left: 4px solid #93c5fd; padding-left: 16px; color: #475569; }}
  </style>
</head>
<body>{body}</body>
</html>"""
    path = report_dir / f"{slugify(title)}.html"
    path.write_text(html, encoding="utf-8")
    return path


def _build_docx_document_xml(markdown_text: str) -> str:
    paragraphs: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        style = ""
        text = line
        if line.startswith("# "):
            style = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            text = line[2:].strip()
        elif line.startswith("## "):
            style = '<w:pPr><w:pStyle w:val="Heading2"/></w:pPr>'
            text = line[3:].strip()
        elif line.startswith("### "):
            style = '<w:pPr><w:pStyle w:val="Heading3"/></w:pPr>'
            text = line[4:].strip()

        safe_text = escape(text)
        paragraphs.append(f"<w:p>{style}<w:r><w:t xml:space=\"preserve\">{safe_text}</w:t></w:r></w:p>")

    body = "".join(paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}"
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
        "</w:body></w:document>"
    )


def _build_docx_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:sz w:val="24"/>
      <w:lang w:val="zh-CN"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="36"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="30"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="26"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def export_docx(report_dir: Path, title: str, markdown_text: str) -> Path:
    path = report_dir / f"{slugify(title)}.docx"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
""",
        )
        archive.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>AI Report Writer</Application>
</Properties>
""",
        )
        archive.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>AI Report Writer</dc:creator>
</cp:coreProperties>
""",
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
""",
        )
        archive.writestr("word/styles.xml", _build_docx_styles_xml())
        archive.writestr("word/document.xml", _build_docx_document_xml(markdown_text))
    return path


def _report_contexts(markdown_text: str, citation_id: str) -> list[str]:
    contexts: list[str] = []
    for paragraph in markdown_paragraphs(markdown_text):
        if citation_id in paragraph:
            contexts.append(truncate(paragraph, 320))
    return contexts[:8]


def _trace_excerpt(chunk: ChunkRecord, report_contexts: list[str]) -> str:
    text = chunk.text.strip()
    if not text:
        return ""

    keywords = re.findall(r"[A-Z]{2,}(?:-[A-Z]{2,})?|[A-Za-z][A-Za-z0-9-]{3,}|[\u4e00-\u9fff]{2,}", " ".join(report_contexts))
    seen: set[str] = set()
    ordered_keywords: list[str] = []
    for keyword in sorted(keywords, key=len, reverse=True):
        lowered = keyword.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered_keywords.append(keyword)

    lowered_text = text.lower()
    for keyword in ordered_keywords:
        position = lowered_text.find(keyword.lower())
        if position != -1:
            start = max(0, position - 180)
            end = min(len(text), position + 420)
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."
            return excerpt

    return truncate(text, 400)


def export_trace(
    report_dir: Path,
    title: str,
    markdown_text: str,
    cited_chunks: list[ChunkRecord],
    validation: ValidationReport,
    generation: dict | None = None,
) -> Path:
    citation_order = collect_citation_ids(markdown_text)
    chunk_map = {chunk.chunk_id: chunk for chunk in cited_chunks}
    trace = [
        {
            "citation_id": citation_id,
            "doc_id": chunk_map[citation_id].doc_id,
            "file_name": chunk_map[citation_id].file_name,
            "page_start": chunk_map[citation_id].page_start,
            "page_end": chunk_map[citation_id].page_end,
            "chunk_type": chunk_map[citation_id].chunk_type,
            "section_title": chunk_map[citation_id].section_title,
            "chunk_position": chunk_map[citation_id].position,
            "source_kind": "primary_pdf" if chunk_map[citation_id].file_name.lower().endswith(".pdf") else "secondary_docx",
            "report_contexts": _report_contexts(markdown_text, citation_id),
            "excerpt": _trace_excerpt(chunk_map[citation_id], _report_contexts(markdown_text, citation_id)),
        }
        for citation_id in citation_order
        if citation_id in chunk_map
    ]
    payload = {
        "title": title,
        "generation": generation or {},
        "validation": validation.model_dump(mode="json"),
        "trace": trace,
    }
    path = report_dir / f"{slugify(title)}-trace.json"
    dump_json(payload, path)
    return path
