from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from PIL import Image
from pydantic import BaseModel, Field
from vllm import SamplingParams

from concordia.concordia.contrib.language_models.vllm.vllm_model import (
    VLLMLanguageModel,
)
from concordia.hdb_simulation.models.schemas.policy.schema import PolicyPage
from concordia.hdb_simulation.models.schemas.policy.schema import PolicyType


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "policies" / "hdb_resale_policy_2023.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_PATH
DEFAULT_AUDIT_PATH = REPO_ROOT / "data" / "policies" / "hdb_resale_policy_2023.audit.jsonl"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "sealion4"
MAX_INPUT_CHARS = 20_000
CLASSIFICATION_MAX_TOKENS = 900
SUMMARY_MAX_TOKENS = 1_200
PDF_EXTRACTION_MAX_TOKENS = 1_500
PDF_RENDER_DPI = 180
DEFAULT_PDF_MAX_PAGES = 0
HTML_EXTRACTION_INPUT_MAX_CHARS = 40_000
TRUNCATION_MARKER_PATTERN = re.compile(
    r"<!-- Extraction truncated after (?P<extracted>\d+) of (?P<total>\d+) pages\. -->"
)


class ClassificationResult(BaseModel):
    reasoning_steps: list[str] = Field(default_factory=list)
    tags: list[PolicyType] = Field(default_factory=list)


class ClassificationAuditRecord(BaseModel):
    path: str
    source: str
    status: str
    tags: list[PolicyType] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    text_characters: int = 0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich a policy JSONL with tags and markdown summaries."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="PolicyPage JSONL to read.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the enriched PolicyPage JSONL.",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="Audit JSONL for classification reasoning.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local VLM path for vLLM.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on number of records to process.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute tags and summary even if a record already has values.",
    )
    parser.add_argument(
        "--pdf-max-pages",
        type=int,
        default=DEFAULT_PDF_MAX_PAGES,
        help="Maximum PDF pages to extract. Use 0 or a negative value to extract all pages.",
    )
    return parser.parse_args()


def initialise_model(args: argparse.Namespace) -> VLLMLanguageModel:
    return VLLMLanguageModel(
        model_name=str(args.model_path.resolve()),
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=16384,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
    )


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def canonical_page_path(record_path: str) -> Path:
    path = Path(record_path)

    if "/raw_html/" in record_path:
        return Path(record_path.replace("/raw_html/", "/page/")).with_suffix(".extracted.md")
    if "/raw_pdf/" in record_path:
        return Path(record_path.replace("/raw_pdf/", "/page/")).with_suffix(".extracted.md")
    if "/page/" in record_path:
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            return path
        return path.with_suffix(".extracted.md")
    return path.with_suffix(".extracted.md")


def canonical_page_record_path(record_path: str) -> str:
    return canonical_page_path(record_path).as_posix()


def source_candidates_for_page_path(page_path: Path) -> list[Path]:
    candidates: list[Path] = []
    page_path_str = page_path.as_posix()

    if page_path.name.endswith(".extracted.md"):
        base_name = page_path.name[: -len(".extracted.md")]
        html_base = Path(page_path_str.replace("/page/", "/raw_html/")).with_name(base_name)
        pdf_base = Path(page_path_str.replace("/page/", "/raw_pdf/")).with_name(base_name)
        candidates.extend(
            [
                html_base.with_suffix(".html"),
                html_base.with_suffix(".htm"),
                pdf_base.with_suffix(".pdf"),
            ]
        )
        return candidates

    candidates.append(Path(page_path_str.replace("/page/", "/raw_html/")))
    candidates.append(Path(page_path_str.replace("/page/", "/raw_pdf/")))
    return candidates


def resolve_source_path(record_path: str) -> Path | None:
    path = Path(record_path)
    candidate_paths = [path]

    if "/page/" in record_path:
        candidate_paths.extend(source_candidates_for_page_path(path))

    for candidate in candidate_paths:
        resolved = resolve_repo_path(candidate)
        if resolved.exists():
            return resolved
    return None


def clean_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        collapsed = " ".join(raw_line.split())
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines).strip()


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "form"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    nodes = main.find_all(["h1", "h2", "h3", "h4", "p", "li"])
    if nodes:
        text = "\n".join(" ".join(node.get_text(" ", strip=True).split()) for node in nodes)
        return clean_text(text)
    return clean_text(main.get_text("\n", strip=True))


def extracted_markdown_cache_path(local_path: Path) -> Path:
    local_path_str = local_path.as_posix()
    if "/raw_html/" in local_path_str:
        return Path(local_path_str.replace("/raw_html/", "/page/")).with_suffix(".extracted.md")
    if "/raw_pdf/" in local_path_str:
        return Path(local_path_str.replace("/raw_pdf/", "/page/")).with_suffix(".extracted.md")
    if local_path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return local_path
    return local_path.with_suffix(".extracted.md")


def truncate_for_prompt(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n<!-- Truncated for extraction prompt length. -->"


def collect_html_fragments(node: Any, fragments: list[str]) -> None:
    if isinstance(node, dict):
        value = node.get("value")
        if isinstance(value, str):
            stripped = value.strip()
            if any(
                tag in stripped.lower()
                for tag in ("<p", "<div", "<table", "<ul", "<ol", "<li", "<h1", "<h2", "<h3", "<br")
            ):
                fragments.append(stripped)
        for child in node.values():
            collect_html_fragments(child, fragments)
        return

    if isinstance(node, list):
        for child in node:
            collect_html_fragments(child, fragments)


def extract_relevant_html_for_llm(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    sections: list[str] = []

    if soup.title and soup.title.string:
        sections.append(f"<title>{soup.title.string.strip()}</title>")

    meta_description = soup.find("meta", attrs={"name": "description"})
    if meta_description and meta_description.get("content"):
        sections.append(
            f'<meta name="description" content="{meta_description["content"].strip()}">'
        )

    next_data = soup.find("script", id="__NEXT_DATA__", attrs={"type": "application/json"})
    if next_data and next_data.string:
        try:
            payload = json.loads(next_data.string)
        except json.JSONDecodeError:
            payload = None

        if payload is not None:
            route_fields = (
                payload.get("props", {})
                .get("pageProps", {})
                .get("layoutData", {})
                .get("sitecore", {})
                .get("route", {})
                .get("fields", {})
            )
            for key in ("pageTitle", "navigationTitle", "publishedDate", "metaDescription"):
                value = route_fields.get(key, {}).get("value")
                if isinstance(value, str) and value.strip():
                    sections.append(f"<{key}>{value.strip()}</{key}>")

            fragments: list[str] = []
            collect_html_fragments(payload, fragments)
            seen_fragments: set[str] = set()
            for fragment in fragments:
                normalised = fragment.strip()
                if normalised and normalised not in seen_fragments:
                    sections.append(normalised)
                    seen_fragments.add(normalised)

    if not sections:
        for tag in soup(["script", "style", "noscript", "svg", "form", "footer", "header", "nav"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        sections.append(str(main))

    combined = "\n\n".join(section.strip() for section in sections if section and section.strip())
    return truncate_for_prompt(combined, max_chars=HTML_EXTRACTION_INPUT_MAX_CHARS)


def build_html_extraction_prompt(page: PolicyPage, html_fragment: str) -> str:
    return f"""You are converting a Singapore HDB policy webpage from HTML into clean markdown.

## Source
- URL: `{page.source}`
- Local path: `{page.path}`

## Task
Convert the relevant policy content from the HTML into markdown.

## Rules
- Preserve headings, numbered steps, bullet lists, tables, links, dates, thresholds, grant amounts, waiting periods, and eligibility rules.
- Preserve the original reading order of the page.
- Keep letter-style content such as salutations, sign-offs, and named signatories when present.
- Exclude website chrome such as global navigation, breadcrumbs, menus, search UI, chatbot widgets, advisory banners, footers, and duplicate boilerplate.
- Do not summarise, paraphrase, or invent missing text.
- Do not include any instructions, notes, or reasoning in the output. Only return the markdown content.
- If the HTML is malformed or a section is incomplete, keep the faithful content that is available.
- Return markdown only. Do not wrap the answer in code fences.

## HTML
```html
{html_fragment}
```
"""


def render_pdf_to_images(
    pdf_path: Path,
    *,
    dpi: int,
    max_pages: int,
) -> tuple[list[Image.Image], int]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "PDF extraction requires PyMuPDF (`pip install pymupdf`)."
        ) from exc

    document = fitz.open(pdf_path)
    try:
        total_pages = len(document)
        page_limit = total_pages if max_pages <= 0 else min(total_pages, max_pages)
        if page_limit == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        images: list[Image.Image] = []
        for page_index in range(page_limit):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )
            images.append(image)
    finally:
        document.close()

    return images, total_pages


def build_pdf_extraction_prompt(
    page: PolicyPage,
    *,
    page_number: int,
    total_pages: int,
) -> str:
    return f"""You are extracting a Singapore HDB policy PDF into structured markdown.

## Source
- URL: `{page.source}`
- Local path: `{page.path}`
- Page: {page_number} of {total_pages}

## Task
Read the attached PDF page image and transcribe it into clean markdown.

## Rules
- Preserve headings, numbered steps, bullet lists, notes, captions, dates, thresholds, grant amounts, waiting periods, and eligibility rules.
- Preserve tables as markdown tables when possible.
- Keep the original reading order for the page.
- Do not summarise, paraphrase, or infer missing text.
- Skip obvious decorative elements only when they add no policy meaning.
- If part of the page is unreadable, note that briefly instead of hallucinating.
- Return markdown only. Do not wrap the answer in code fences.
"""


def normalise_markdown_response(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:markdown)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def cached_pdf_covers_request(markdown: str, *, max_pages: int) -> bool:
    match = TRUNCATION_MARKER_PATTERN.search(markdown)
    if match is None:
        return True

    extracted_pages = int(match.group("extracted"))
    total_pages = int(match.group("total"))
    target_pages = total_pages if max_pages <= 0 else min(total_pages, max_pages)
    return extracted_pages >= target_pages


def sample_pdf_markdown_with_vllm(
    model: VLLMLanguageModel,
    *,
    prompt: str,
    image: Image.Image,
    max_tokens: int,
) -> str:
    engine = getattr(model, "_llm", None)
    if engine is None or not hasattr(engine, "chat"):
        raise RuntimeError(
            "The configured VLLM model does not expose `LLM.chat`, which is required "
            "for multimodal PDF extraction."
        )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_pil", "image_pil": image},
            ],
        }
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    outputs = engine.chat(
        conversation,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return normalise_markdown_response(outputs[0].outputs[0].text)


def sample_markdown_with_text_model(
    model: VLLMLanguageModel,
    *,
    prompt: str,
) -> str:
    return normalise_markdown_response(model.sample_text(prompt))


def extract_markdown_from_html(
    page: PolicyPage,
    model: VLLMLanguageModel,
) -> str:
    source_path = resolve_source_path(page.path)
    if source_path is None:
        raise FileNotFoundError(f"Could not resolve local policy file for path: {page.path}")

    cache_path = resolve_repo_path(canonical_page_path(page.path))
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    html = source_path.read_text(encoding="utf-8")
    html_fragment = extract_relevant_html_for_llm(html)
    markdown = sample_markdown_with_text_model(
        model,
        prompt=build_html_extraction_prompt(page, html_fragment),
    ).strip()

    cache_path.write_text(markdown + "\n", encoding="utf-8")
    return markdown


def extract_markdown_from_pdf(
    page: PolicyPage,
    model: VLLMLanguageModel,
    *,
    max_pages: int,
) -> str:
    source_path = resolve_source_path(page.path)
    if source_path is None:
        raise FileNotFoundError(f"Could not resolve local policy file for path: {page.path}")

    cache_path = resolve_repo_path(canonical_page_path(page.path))
    if cache_path.exists():
        cached_markdown = cache_path.read_text(encoding="utf-8")
        if cached_pdf_covers_request(cached_markdown, max_pages=max_pages):
            return cached_markdown

    images, total_pages = render_pdf_to_images(
        source_path,
        dpi=PDF_RENDER_DPI,
        max_pages=max_pages,
    )

    extracted_pages: list[str] = []
    for page_number, image in enumerate(images, start=1):
        prompt = build_pdf_extraction_prompt(
            page,
            page_number=page_number,
            total_pages=total_pages,
        )
        markdown = sample_pdf_markdown_with_vllm(
            model,
            prompt=prompt,
            image=image,
            max_tokens=PDF_EXTRACTION_MAX_TOKENS,
        )
        extracted_pages.append(f"<!-- Page {page_number}/{total_pages} -->\n{markdown}")

    if total_pages > len(images):
        extracted_pages.append(
            f"<!-- Extraction truncated after {len(images)} of {total_pages} pages. -->"
        )

    combined_markdown = "\n\n".join(part.strip() for part in extracted_pages if part.strip()).strip()
    cache_path.write_text(combined_markdown + "\n", encoding="utf-8")
    return combined_markdown


def load_policy_content(
    page: PolicyPage,
    model: VLLMLanguageModel,
    *,
    pdf_max_pages: int,
) -> str:
    page_markdown_path = resolve_repo_path(canonical_page_path(page.path))
    if page_markdown_path.exists():
        return clean_text(page_markdown_path.read_text(encoding="utf-8"))

    source_path = resolve_source_path(page.path)
    if source_path is None:
        raise FileNotFoundError(f"Could not resolve local policy file for path: {page.path}")

    suffix = source_path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return extract_markdown_from_html(page, model)
    if suffix in {".md", ".txt"}:
        return clean_text(source_path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return extract_markdown_from_pdf(page, model, max_pages=pdf_max_pages)
    raise ValueError(f"Unsupported local file type for summarisation: {source_path.suffix}")


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Truncated for prompt length.]"


def build_classification_prompt(page: PolicyPage, article_text: str) -> str:
    allowed_tags = "\n".join(f"- {policy_type.value}" for policy_type in PolicyType)
    return f"""# Role

You classify Singapore HDB resale policy documents.

## Objective

1. Read the extracted content.
2. Decide which policy types apply.
3. More than one tag may apply.
4. Write short reasoning steps grounded in the content.

## Allowed Policy Types

{allowed_tags}

## Document Metadata

- Source: `{page.source}`
- Path: `{page.path}`

## Extracted Content

```text
{article_text}
```

## Output Format

Return JSON only. Do not use markdown fences.

```json
{{
  "reasoning_steps": [
    "step 1",
    "step 2"
  ],
  "tags": [
    "exact policy type value"
  ]
}}
```

## Rules

- Use only the allowed policy type values.
- If no tag is strongly supported, return an empty `tags` list.
- Keep `reasoning_steps` concise and evidence-based.
"""


def build_summary_prompt(page: PolicyPage, article_text: str, tags: list[PolicyType]) -> str:
    tag_text = ", ".join(tag.value for tag in tags) if tags else "No tags assigned"
    return f"""# Role

You summarise Singapore HDB resale policy documents.

## Inputs

- Source: `{page.source}`
- Assigned policy tags: {tag_text}

## Extracted Content

```text
{article_text}
```

## Required Output

Write markdown with exactly these sections:

# Policy Summary
## Scope
## Who It Applies To
## Key Rules
## Implications
## Process and Timing
## Exceptions and Edge Cases
## Practical Takeaways

## Requirements

- Use only facts supported by the extracted content.
- If a section is not specified, write `Not specified in this document.`
- Be concise but specific.
- Include thresholds, grant amounts, eligibility conditions, dates, waiting periods, and procedural steps when stated.
- Do not include any information that is not explicitly supported by the extracted content.
- Do not include any text before or after the markdown summary.
- Do not include ANY internal reasoning steps or notes, or instructions in the output. Only the markdown summary should be returned.
"""


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in model output.", text, 0)


def decode_jsonl_line(raw_line: str) -> list[Any]:
    decoder = json.JSONDecoder()
    records: list[Any] = []
    text = raw_line.strip()
    index = 0

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        record, index = decoder.raw_decode(text, idx=index)
        records.append(record)

    return records


def call_model_for_json(
    model: VLLMLanguageModel,
    prompt: str,
    result_model: type[BaseModel],
    *,
    max_tokens: int,
    retries: int = 3,
) -> BaseModel:
    last_error: Exception | None = None
    for _ in range(retries):
        response = model.sample_text(
            prompt,
            max_tokens=max_tokens,
            json_schema=result_model.model_json_schema(),
        )
        try:
            payload = extract_json_object(response)
            return result_model.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            prompt = (
                prompt
                + "\n\nYour previous response was not valid for the requested JSON shape. "
                + "Return JSON only and follow the schema exactly."
            )
    raise RuntimeError(
        f"Failed to parse model response as JSON after {retries} attempts: {last_error}"
    )


def load_policy_pages(input_path: Path) -> list[PolicyPage]:
    pages: list[PolicyPage] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record_number = 1
            try:
                for record_number, record in enumerate(
                    decode_jsonl_line(stripped),
                    start=1,
                ):
                    pages.append(PolicyPage.model_validate(record))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "Invalid PolicyPage record "
                    + f"at line {line_number}, record {record_number} in {input_path}: {exc}"
                ) from exc
    return pages


def dump_jsonl(records: list[BaseModel], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def append_jsonl(records: list[BaseModel], output_path: Path) -> None:
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def enrich_page(
    page: PolicyPage,
    model: VLLMLanguageModel,
    *,
    pdf_max_pages: int,
) -> tuple[PolicyPage, ClassificationAuditRecord]:
    canonical_path = canonical_page_record_path(page.path)
    article_text = truncate_text(
        load_policy_content(page, model, pdf_max_pages=pdf_max_pages),
        MAX_INPUT_CHARS,
    )

    classification = call_model_for_json(
        model,
        build_classification_prompt(page, article_text),
        ClassificationResult,
        max_tokens=CLASSIFICATION_MAX_TOKENS,
    )
    classification = ClassificationResult.model_validate(classification)

    markdown_summary = model.sample_text(
        build_summary_prompt(page, article_text, classification.tags),
        max_tokens=SUMMARY_MAX_TOKENS,
    ).strip()

    updated_page = PolicyPage(
        path=canonical_path,
        source=page.source,
        summary=markdown_summary,
        tags=classification.tags,
    )
    audit_record = ClassificationAuditRecord(
        path=canonical_path,
        source=page.source,
        status="ok",
        tags=classification.tags,
        reasoning_steps=classification.reasoning_steps,
        text_characters=len(article_text),
    )
    return updated_page, audit_record


def should_process(page: PolicyPage, overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return True
    return not page.summary.strip() and not page.tags


def main() -> None:
    args = parse_args()

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    audit_path = args.audit_path.resolve() if args.audit_path else None

    pages = load_policy_pages(input_path)
    if args.max_records is not None:
        pages = pages[: args.max_records]

    model = initialise_model(args)

    enriched_pages: list[PolicyPage] = []
    audit_records: list[ClassificationAuditRecord] = []

    for index, page in enumerate(pages, start=1):
        print(f"[{index}/{len(pages)}] Processing {page.source}")
        canonical_path = canonical_page_record_path(page.path)
        if not should_process(page, args.overwrite_existing):
            enriched_pages.append(
                PolicyPage(
                    path=canonical_path,
                    source=page.source,
                    summary=page.summary,
                    tags=page.tags,
                )
            )
            audit_records.append(
                ClassificationAuditRecord(
                    path=canonical_path,
                    source=page.source,
                    status="skipped_existing",
                    tags=page.tags,
                    text_characters=0,
                )
            )
            continue

        try:
            enriched_page, audit_record = enrich_page(
                page,
                model,
                pdf_max_pages=args.pdf_max_pages,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")
            enriched_pages.append(
                PolicyPage(
                    path=canonical_path,
                    source=page.source,
                    summary=page.summary,
                    tags=page.tags,
                )
            )
            audit_records.append(
                ClassificationAuditRecord(
                    path=canonical_path,
                    source=page.source,
                    status="error",
                    tags=page.tags,
                    text_characters=0,
                    error=str(exc),
                )
            )
            continue

        print(f"  tags: {[tag.value for tag in enriched_page.tags]}")
        enriched_pages.append(enriched_page)
        audit_records.append(audit_record)

    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dump_jsonl(enriched_pages, temp_output_path)
    temp_output_path.replace(output_path)

    if audit_path is not None:
        append_jsonl(audit_records, audit_path)

    print(f"Wrote {len(enriched_pages)} policy records to {output_path}")
    if audit_path is not None:
        print(f"Wrote {len(audit_records)} audit records to {audit_path}")


if __name__ == "__main__":
    main()
