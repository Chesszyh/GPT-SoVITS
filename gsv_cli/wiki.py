from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


WIKI_LINK_RE = re.compile(r"(\]\()([^)]+)(\))")
REPO_WIKI_PREFIXES = (
    "/RVC-Boss/GPT-SoVITS/",
    "/Chesszyh/GPT-SoVITS/",
)
LANGUAGE_DIRS = ("en", "zh")


def publish_github_wiki_source(source_dir: Path | str, target_dir: Path | str) -> None:
    source = Path(source_dir)
    target = Path(target_dir)
    slug_map = _collect_slug_map(source)

    _clear_target(target)
    _copy_root_page(source, target, "Home.md", slug_map)
    _copy_root_page(source, target, "_Sidebar.md", slug_map)

    for lang in LANGUAGE_DIRS:
        lang_dir = source / lang
        for source_page in sorted(lang_dir.rglob("*.md")):
            slug = source_page.stem
            target_page = target / f"{lang}-{slug}.md"
            content = source_page.read_text(encoding="utf-8")
            target_page.write_text(_rewrite_wiki_links(content, lang, slug_map), encoding="utf-8")


def _copy_root_page(source: Path, target: Path, name: str, slug_map: dict[str, set[str]]) -> None:
    source_page = source / name
    if not source_page.exists():
        return
    content = source_page.read_text(encoding="utf-8")
    (target / name).write_text(_rewrite_wiki_links(content, None, slug_map), encoding="utf-8")


def _collect_slug_map(source: Path) -> dict[str, set[str]]:
    slug_map: dict[str, set[str]] = {}
    for lang in LANGUAGE_DIRS:
        lang_dir = source / lang
        slugs = [page.stem for page in sorted(lang_dir.rglob("*.md"))]
        duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
        if duplicates:
            joined = ", ".join(duplicates)
            raise ValueError(f"Duplicate wiki page slugs under {lang_dir}: {joined}")
        slug_map[lang] = set(slugs)
    return slug_map


def _clear_target(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _rewrite_wiki_links(content: str, current_lang: str | None, slug_map: dict[str, set[str]]) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, target, suffix = match.groups()
        rewritten = _rewrite_link_target(target, current_lang, slug_map)
        return f"{prefix}{rewritten}{suffix}"

    return WIKI_LINK_RE.sub(replace, content)


def _rewrite_link_target(target: str, current_lang: str | None, slug_map: dict[str, set[str]]) -> str:
    if _is_external_or_anchor_link(target):
        return target

    path, anchor = _split_anchor(target)
    normalized = path[:-3] if path.endswith(".md") else path

    explicit_lang_target = _rewrite_explicit_language_target(normalized, anchor, slug_map)
    if explicit_lang_target is not None:
        return explicit_lang_target

    repo_target = _rewrite_repo_wiki_target(normalized, anchor, current_lang, slug_map)
    if repo_target is not None:
        return repo_target

    if current_lang is not None:
        slug = Path(normalized).name
        if slug in slug_map[current_lang]:
            return _with_anchor(f"{current_lang}-{slug}", anchor)

    return target


def _is_external_or_anchor_link(target: str) -> bool:
    return (
        target.startswith("#")
        or target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
    )


def _split_anchor(target: str) -> tuple[str, str]:
    if "#" not in target:
        return target, ""
    path, anchor = target.split("#", 1)
    return path, f"#{anchor}"


def _rewrite_explicit_language_target(
    normalized: str, anchor: str, slug_map: dict[str, set[str]]
) -> str | None:
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2 or parts[0] not in LANGUAGE_DIRS:
        return None
    lang = parts[0]
    slug = parts[-1]
    if slug not in slug_map[lang]:
        return None
    return _with_anchor(f"{lang}-{slug}", anchor)


def _rewrite_repo_wiki_target(
    normalized: str, anchor: str, current_lang: str | None, slug_map: dict[str, set[str]]
) -> str | None:
    for prefix in REPO_WIKI_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        slug = normalized.removeprefix(prefix).strip("/").split("/")[-1]
        if current_lang is not None and slug in slug_map[current_lang]:
            return _with_anchor(f"{current_lang}-{slug}", anchor)
        return None
    return None


def _with_anchor(target: str, anchor: str) -> str:
    return f"{target}{anchor}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish repository wiki source into a GitHub wiki tree.")
    parser.add_argument("--source", default="wiki", type=Path)
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    publish_github_wiki_source(args.source, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
