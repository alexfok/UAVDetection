from __future__ import annotations

from collections.abc import Iterable, Mapping


def normalise_label_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    if not aliases:
        return {}
    return {
        str(source).strip().lower(): str(target).strip()
        for source, target in aliases.items()
        if str(source).strip() and str(target).strip()
    }


def display_label(label: str, aliases: Mapping[str, str] | None) -> str:
    normalised = normalise_label_aliases(aliases)
    return normalised.get(label.strip().lower(), label)


def resolve_target_class_ids(
    names: Mapping[int, str],
    targets: Iterable[str],
    aliases: Mapping[str, str] | None,
) -> list[int] | None:
    target_names = {name.strip().lower() for name in targets if name and name.strip()}
    if not target_names:
        return None

    normalised_aliases = normalise_label_aliases(aliases)
    ids: list[int] = []
    for class_id, raw_name in names.items():
        raw_lower = raw_name.lower()
        aliased_lower = normalised_aliases.get(raw_lower, raw_name).lower()
        if raw_lower in target_names or aliased_lower in target_names:
            ids.append(class_id)
    return ids or None

