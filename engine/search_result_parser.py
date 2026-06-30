"""search-result-parser - Parse GitCode model search output to Markdown tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchQuerySummary:
    """Parsed query summary metadata."""
    query_number: int
    model_name: Optional[str]
    attribute_tag: Optional[str]
    framework: Optional[str]
    hardware: Optional[str]
    total_matches: int
    showing_count: int


@dataclass
class ModelSearchResult:
    """Parsed model search result entry."""
    model_name: str
    is_official: bool
    framework: Optional[str]
    hardware: Optional[str]
    repository: Optional[str]
    link: Optional[str]

    @property
    def category(self) -> str:
        """Return Chinese category label."""
        return "官方文档" if self.is_official else "开发者实践"

    def to_row(self, include_repository: bool = True) -> list[str]:
        """Convert to table row values."""
        if include_repository:
            # Developer practice: Model Name shows repository, no separate Repository column
            row = [
                self.repository or "-",
                self.framework or "-",
                self.hardware or "-",
            ]
        else:
            # Official docs
            row = [
                self.model_name,
                self.framework or "-",
                self.hardware or "-",
            ]
        row.append(_format_link(self.link))
        return row


def _format_link(link: Optional[str]) -> str:
    """Format link as 'Link' hyperlink."""
    if not link:
        return "-"
    return f"[Link]({link})"


def _truncate(text: str, max_width: int = 20) -> str:
    """Truncate text with ellipsis if needed."""
    if not text:
        return "-"
    if len(text) <= max_width:
        return text
    return text[:max_width - 3] + "..."


def _wrap_on_slash(text: Optional[str], max_width: int = 28) -> str:
    """Format text wrapping only on slashes, returning '-' for None/empty."""
    if not text:
        return "-"
    if len(text) <= max_width:
        return text
    # Build lines by splitting on slash only
    parts = text.split('/')
    lines = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_width:
            current = f"{current}/{part}"
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return '\n'.join(lines)


def _wrap_url(link: Optional[str], max_width: int = 35) -> str:
    """Format URL with wrapping on slashes, returning '-' for None/empty."""
    if not link:
        return "-"
    if '://' not in link:
        return _wrap_on_slash(link, max_width)
    # Extract domain and path
    domain_end = link.find('://') + 3
    domain = link[:domain_end]
    path = link[domain_end:]
    if not path:
        return domain
    # If domain alone exceeds max_width, truncate it
    if len(domain) > max_width - 3:
        return domain[:max_width - 3] + "..."
    # Build lines
    lines = [domain]
    current = ""
    for part in path.split('/'):
        if not part:
            continue
        test = f"{current}/{part}" if current else part
        if len(test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = part
            while len(current) > max_width:
                lines.append(current[:max_width])
                current = current[max_width:]
    if current:
        lines.append(current)
    return '\n'.join(lines)


def parse_query_summary(line: str) -> SearchQuerySummary:
    """Extract metadata from the query summary line. Supports both English and Chinese formats."""
    # Query number: English [query N] or Chinese 【查询N】
    query_match = re.search(r'\[query\s+(\d+)\]', line) or re.search(r'【查询(\d+)】', line)
    query_number = int(query_match.group(1)) if query_match else 1

    # Total matches: English "-> N matches" or Chinese "命中 N 个"
    matches_match = re.search(r'->\s*(\d+)\s*matches', line) or re.search(r'命中\s*(\d+)\s*个', line)
    total_matches = int(matches_match.group(1)) if matches_match else 0

    # Showing count: English "showing top N" or Chinese "仅展示前 N 个"
    showing_match = re.search(r'showing\s+top\s+(\d+)', line) or re.search(r'仅展示前\s*(\d+)\s*个', line)
    showing_count = int(showing_match.group(1)) if showing_match else 0

    # Model name: English "model-name:" or Chinese "model-name:"
    model_name_match = re.search(r'model-name:\s*([^\s]+)', line)
    model_name = model_name_match.group(1) if model_name_match else None

    attr_match = re.search(r'attribute-tag:\s*([^\s]+)', line)
    attribute_tag = attr_match.group(1) if attr_match else None

    framework_match = re.search(r'f?ramwork:\s*([^\s]+)', line)
    framework = framework_match.group(1) if framework_match else None

    hardware_match = re.search(r'hardware:\s*([^\s]+)', line)
    hardware = hardware_match.group(1) if hardware_match else None

    return SearchQuerySummary(
        query_number=query_number,
        model_name=model_name,
        attribute_tag=attribute_tag,
        framework=framework,
        hardware=hardware,
        total_matches=total_matches,
        showing_count=showing_count,
    )


def parse_result_entry(entry_text: str) -> ModelSearchResult:
    """Parse a single model result entry. Supports both English and Chinese formats."""
    # Process each line individually to handle multi-line format properly
    lines = entry_text.strip().split('\n')

    # Try English format first: "Model Name: xxx [official]" or "[official]"
    model_name: Optional[str] = None
    is_official: bool = False
    for line in lines:
        line = line.strip()
        model_match = re.search(r'Model Name:\s*([^\[]+?)\s*\[(official|third-party)\]', line)
        if model_match:
            model_name = model_match.group(1).strip()
            is_official = model_match.group(2) == 'official'
            break
        # Try Chinese format: "模型名：GLM-5【官方】"
        model_match = re.search(r'模型名：\s*([^\[]+?)\s*【(官方|三方)】', line)
        if model_match:
            model_name = model_match.group(1).strip()
            is_official = model_match.group(2) == '官方'
            break

    if model_name is None:
        raise ValueError(f"Cannot parse Model Name from: {entry_text[:100]}")

    # Parse fields - process each line individually
    fields: dict[str, Optional[str]] = {"framework": None, "hardware": None, "repository": None, "link": None}

    # Define field patterns - label followed by value (only match at start of line)
    field_patterns = [
        (r'^适配框架[：:]\s*(.+)$', 'framework'),  # Chinese: vllm-ascend
        (r'^Framework:\s*(.+)$', 'framework'),      # English
        (r'^适配硬件[：:]\s*(.+)$', 'hardware'),  # Chinese: A2,A3
        (r'^Hardware:\s*(.+)$', 'hardware'),      # English
        (r'^仓库[：:]\s*(.+)$', 'repository'),   # Chinese
        (r'^Repository:\s*(.+)$', 'repository'),   # English
        (r'^链接[：:]\s*(.+)$', 'link'),          # Chinese
        (r'^Link:\s*(.+)$', 'link'),              # English
    ]

    for line in lines:
        line_stripped = line.strip()
        for pattern, field_key in field_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                fields[field_key] = match.group(1).strip()
                break

    return ModelSearchResult(
        model_name=model_name,
        is_official=is_official,
        framework=fields["framework"],
        hardware=fields["hardware"],
        repository=fields["repository"],
        link=fields["link"],
    )


def _escape_markdown_cell(text: str) -> str:
    """Escape special characters for Markdown table cell."""
    if not text:
        return ""
    text = text.replace('|', '\\|')
    # Preserve newlines as <br> for wrapped content
    text = text.replace('\n', '<br>')
    return text


def format_markdown_table(
    results: list[ModelSearchResult],
    summary: Optional[SearchQuerySummary] = None,
) -> str:
    """Format search results as grouped Markdown tables."""
    if not results:
        return "No results found."

    official_results = [r for r in results if r.is_official]
    third_party_results = [r for r in results if not r.is_official]

    lines = []

    # Add summary line at the top
    if summary and summary.total_matches > 0:
        lines.append(f"共查询到 {summary.total_matches} 条结果，当前仅展示前 {summary.showing_count} 条")
        lines.append("")

    if official_results:
        lines.append("## 官方文档")
        lines.append("")
        header = "| Model Name | Framework | Hardware | Link |"
        separator = "|---|---|---|---|"
        lines.append(header)
        lines.append(separator)
        for result in official_results:
            row = result.to_row(include_repository=False)
            escaped = [_escape_markdown_cell(v) for v in row]
            lines.append(f"| {' | '.join(escaped)} |")

    if third_party_results:
        if official_results:
            lines.append("")
        lines.append("## 开发者实践")
        lines.append("")
        header = "| Model Name | Framework | Hardware | Link |"
        separator = "|---|---|---|---|"
        lines.append(header)
        lines.append(separator)
        for result in third_party_results:
            row = result.to_row(include_repository=True)
            escaped = [_escape_markdown_cell(v) for v in row]
            lines.append(f"| {' | '.join(escaped)} |")

    return '\n'.join(lines)


def parse_and_format_search_results(raw_output: str) -> str:
    """Parse GitCode model search output and format as grouped Markdown table."""
    if not raw_output or not raw_output.strip():
        return "No search results to parse."

    lines = raw_output.strip().split('\n')

    summary: Optional[SearchQuerySummary] = None
    result_lines: list[str] = []
    in_results = False
    entry_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith('='):
            continue

        # Detect result section - English: "Top X results", Chinese: "前 X 个结果"
        if ('Top' in stripped and 'result' in stripped) or ('个结果' in stripped):
            in_results = True
            continue

        # Parse summary - English: "[query 1]", Chinese: "【查询1】"
        if not summary and ('[query' in stripped or '【查询' in stripped):
            try:
                summary = parse_query_summary(stripped)
            except Exception:
                pass
            continue

        if in_results:
            # Entry can start with English or Chinese marker
            if stripped.startswith('Model Name:') or stripped.startswith('模型名：'):
                if entry_buffer:
                    result_lines.append('\n'.join(entry_buffer))
                entry_buffer = [stripped]
            elif entry_buffer:
                entry_buffer.append(stripped)

    if entry_buffer:
        result_lines.append('\n'.join(entry_buffer))

    results: list[ModelSearchResult] = []
    for entry_text in result_lines:
        try:
            result = parse_result_entry(entry_text)
            results.append(result)
        except Exception:
            continue

    return format_markdown_table(results, summary)
