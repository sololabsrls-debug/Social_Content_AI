import re


def extract_first_name(full_name: str) -> str:
    parts = (full_name or "").strip().split()
    return parts[-1] if parts else full_name


def normalize_campaign_message_template(message: str) -> str:
    normalized = (message or "").strip()
    normalized = re.sub(
        r"\{+\s*nome\s*\}+|\[\s*nome\s*\]|<\s*nome\s*>",
        "{{nome}}",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r'(?i)\b(ciao|gentile|buongiorno|salve|cara|caro)\s+["\'\[\(\{<]*nome["\'\]\)\}>]*',
        lambda match: f"{match.group(1)} {{{{nome}}}}",
        normalized,
        count=1,
    )
    if "{{nome}}" not in normalized:
        normalized = re.sub(
            r'(?i)(^|[\s"\'\[\(\{<])nome(?=$|[\s,!.?;:"\'\]\)\}>])',
            lambda match: f"{match.group(1)}{{{{nome}}}}",
            normalized,
            count=1,
        )
    return normalized


def render_campaign_message(message_template: str, full_name: str) -> str:
    first_name = extract_first_name(full_name)
    normalized = normalize_campaign_message_template(message_template)
    if not first_name:
        return re.sub(r"\s{2,}", " ", normalized.replace("{{nome}}", "")).strip(" ,")
    return normalized.replace("{{nome}}", first_name)
