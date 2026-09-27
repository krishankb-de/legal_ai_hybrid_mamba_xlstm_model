"""Licence register in code (plan P3-B, rule R9). ``Docs/CORPUS_LICENCE_REGISTER.md`` is its prose twin.

Every ``Document.licence`` must be an id of ``LICENCES``. ``require_known`` refuses ``unknown`` and
unregistered ids; ``check_document`` refuses a record that claims more than its licence allows (a
non-commercial licence marked ``commercial_safe``, or a research-only licence without
``research_only``). ``lexhybrid.data.corpus.build.build_corpus`` applies both to every document.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Licence:
    id: str
    name: str
    commercial_safe: bool
    research_only: bool
    attribution: str
    url: str


LICENCES: dict[str, Licence] = {
    lic.id: lic
    for lic in (
        Licence(
            "DE-UrhG-5",
            "Amtliches Werk, § 5 UrhG (statutes, ordinances, court decisions, official guiding principles)",
            commercial_safe=True,
            research_only=False,
            attribution="source named as good practice; no copyright protection",
            url="https://www.gesetze-im-internet.de/urhg/__5.html",
        ),
        Licence(
            "CH-URG-5",
            "Nicht geschütztes Werk, Art. 5 URG (Swiss enactments, decisions and reports of authorities)",
            commercial_safe=True,
            research_only=False,
            attribution="source named as good practice; no copyright protection",
            url="https://www.fedlex.admin.ch/eli/cc/1993/1798_1798_1798/de#art_5",
        ),
        Licence(
            "CC-BY-4.0",
            "Creative Commons Attribution 4.0 International",
            commercial_safe=True,
            research_only=False,
            attribution="name the source and indicate changes",
            url="https://creativecommons.org/licenses/by/4.0/",
        ),
        Licence(
            "ODbL-1.0",
            "Open Data Commons Open Database License 1.0",
            commercial_safe=True,
            research_only=False,
            attribution="attribute the database; derived databases share-alike; produced works carry a notice",
            url="https://opendatacommons.org/licenses/odbl/1-0/",
        ),
        Licence(
            "ODC-By-1.0",
            "Open Data Commons Attribution License 1.0",
            commercial_safe=True,
            research_only=False,
            attribution="attribute the database (FineWeb-2 also asks users to respect CommonCrawl's terms)",
            url="https://opendatacommons.org/licenses/by/1-0/",
        ),
        Licence(
            "CC-BY-NC-SA-4.0",
            "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
            commercial_safe=False,
            research_only=True,
            attribution="attribute; non-commercial only; share-alike",
            url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        ),
        Licence(
            "CC-BY-NC-4.0",
            "Creative Commons Attribution-NonCommercial 4.0 International",
            commercial_safe=False,
            research_only=True,
            attribution="attribute; non-commercial only",
            url="https://creativecommons.org/licenses/by-nc/4.0/",
        ),
        Licence(
            "unknown",
            "Licence not established: refused by the corpus build",
            commercial_safe=False,
            research_only=False,
            attribution="",
            url="",
        ),
    )
}


class LicenceError(ValueError):
    """A document's licence is unknown, unregistered, or inconsistent with its flags."""


def require_known(licence_id: str) -> Licence:
    """The registered licence, or ``LicenceError`` for ``unknown`` and unregistered ids."""
    if licence_id == "unknown":
        raise LicenceError("licence 'unknown' is refused: establish the licence and register it first (R9)")
    if licence_id not in LICENCES:
        raise LicenceError(f"licence {licence_id!r} is not in the register ({sorted(LICENCES)})")
    return LICENCES[licence_id]


def is_commercial_safe(licence_id: str) -> bool:
    """True when the registered licence permits a shippable (commercial) training arm."""
    return require_known(licence_id).commercial_safe


def check_document(doc) -> list[str]:
    """Problems with a document's licence flags; empty when consistent."""
    try:
        lic = require_known(doc.licence)
    except LicenceError as e:
        return [str(e)]
    problems = []
    if doc.commercial_safe and not lic.commercial_safe:
        problems.append(f"{doc.id}: marked commercial_safe under {lic.id}, which does not permit it")
    if lic.research_only and not doc.research_only:
        problems.append(f"{doc.id}: {lic.id} is research-only but the document is not marked research_only")
    return problems
