"""Regression checks for the public review-release metadata."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_review_release_docs_reference_v3_1_review_4() -> None:
    documents = [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "CODE_AVAILABILITY.md",
        REPOSITORY_ROOT / "docs" / "MODEL_ZOO.md",
        REPOSITORY_ROOT / "docs" / "OPEN_SOURCE_DECISIONS.md",
        REPOSITORY_ROOT / "docs" / "RELEASE_PLAN.md",
        REPOSITORY_ROOT / "docs" / "REPRODUCIBILITY.md",
    ]

    for document in documents:
        assert "v3.1-review.4" in document.read_text(encoding="utf-8")


def test_readme_quick_start_lists_all_release_manifests() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## Reviewer Quick Start", 1)[1].split(
        "This evaluates each released checkpoint", 1
    )[0]

    assert "all ten assets" in quick_start
    assert "shasum -a 256 -c CHECKPOINT_SHA256SUMS" in quick_start
    assert "shasum -a 256 -c SIDD_QUICK_EVAL_ZIP_SHA256SUMS" in quick_start


def test_sidd_plus_notice_contains_required_terms_and_sources() -> None:
    notice = (
        REPOSITORY_ROOT / "docs" / "SIDD_PLUS_LICENSE_AND_NOTICE.md"
    ).read_text(encoding="utf-8")

    required_fragments = [
        "open research and educational purposes",
        "copyright of the images remains",
        "not to redistribute the data without this notice",
        "https://abdokamel.github.io/sidd/",
        "https://competitions.codalab.org/competitions/22230",
        "https://competitions.codalab.org/competitions/22231",
    ]
    for fragment in required_fragments:
        assert fragment in notice


def test_sidd_zip_manifest_has_one_sha256_entry() -> None:
    manifest = (
        REPOSITORY_ROOT / "docs" / "SIDD_QUICK_EVAL_ZIP_SHA256SUMS"
    ).read_text(encoding="utf-8").splitlines()

    assert manifest == [
        "ae84d95526999a07d5ef267b3e7eaccb0c889e3c26027c2fa50c852eeb2ea17f  "
        "sidd_quick_eval_256.zip"
    ]
