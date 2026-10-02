"""Document catalog: one entry per source manuscript.

Every chunk in Qdrant is tagged with its document's ``doc_id``. The router reads
this catalog to decide which document(s) a question belongs to, and retrieval is
then restricted to those documents with a payload filter.

To add a manuscript: drop the PDF in ``data/pdfs/``, add an entry below, and
re-run ``python ingest.py``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    file: str
    title: str
    # Names / spellings that unambiguously point at this document.
    aliases: tuple[str, ...]
    # Topics covered; the LLM router reads this for questions without a name.
    description: str


DOCUMENTS: tuple[SourceDoc, ...] = (
    SourceDoc(
        doc_id="abu_bakr",
        file="Hz Abu Bakr RA 4.pdf",
        title="Hz Abu Bakr Siddique RA",
        aliases=(
            "abu bakr", "abu bakar", "abubakr", "abubakar", "abu baker",
            "siddiq", "siddique", "siddeeq", "as siddiq", "first caliph",
        ),
        description=(
            "Life and caliphate of Abu Bakr as-Siddiq (11-13 AH): lineage, family, "
            "personality, assumption of the caliphate after the Prophet's death, "
            "Saqifah, the delayed pledge of Ali, the incident of Qirtas, his first "
            "address, the army of Usama bin Zayd, the Ridda (apostasy) wars, the false "
            "prophets (Musaylimah, Tulayha, Sajah), the Zakat rejectors and Battle of "
            "Dhul Qissah, compilation/preservation of the Quran, Khalid bin Walid's "
            "march to Hira, Abu Ubaidah's march to Syria, Battle of Ajnadayn, "
            "appointment of Umar as successor, his death and advice to Umar. Also an "
            "introduction to why the Rightly Guided Caliphs are studied."
        ),
    ),
    SourceDoc(
        doc_id="umar",
        file="Hz Umer Farooq RA - Session 4.pdf",
        title="Hz Umer Farooq RA",
        aliases=(
            "umar", "umer", "omar", "farooq", "faruq", "farouk", "al faruq",
            "ibn al khattab", "bin al khattab", "second caliph",
        ),
        description=(
            "Life and caliphate of Umar ibn al-Khattab (13-23 AH): lineage, tribe "
            "Banu Adi, physical description, acceptance of Islam, revelations "
            "confirming his views, conquests (fall of Damascus, Battle of Yarmouk, "
            "fall of Jerusalem, Plague of Amwas, conquest of Egypt and Alexandria, "
            "fall of the Sassanids, Qadisiyyah, Battle of the Bridge), administration: "
            "shura, night patrols, justice, provinces, communal land, the Diwan "
            "registry, economic policy and zakat, judicial system, Tarawih, standing "
            "army, and his assassination and death."
        ),
    ),
    SourceDoc(
        doc_id="uthman",
        file="Hz Uthman RA-1.pdf",
        title="Hz Uthman ibn Affan RA",
        aliases=(
            "uthman", "usman", "osman", "othman", "usmaan", "ibn affan",
            "bin affan", "dhun nurayn", "dhun noorayn", "zun noorain",
            "third caliph",
        ),
        description=(
            "Life and caliphate of Uthman ibn Affan (23-35 AH): lineage, Banu "
            "Umayyah, virtues, Dhun-Noorayn, embracing Islam, the shura after Umar's "
            "death and his election, the inherited empire, Kufah, Azerbaijan and "
            "Armenia, death of the last Persian king, Alexandria, conquest of North "
            "Africa, Cyprus and the first naval campaign, Umm Haram, the official "
            "Mushaf of the Quran, extension of Masjid an-Nabawi, the causes of the "
            "fitnah, Abu Dharr controversy, Abdullah ibn Saba, the eleven complaints "
            "and rebuttals, the forged letter, the siege of Madinah, and his "
            "martyrdom in 35 AH."
        ),
    ),
    SourceDoc(
        doc_id="ali",
        file="Ali ibn Abi Talib RA V3.pdf",
        title="Hz Ali ibn Abi Talib RA",
        aliases=(
            "ali", "ali ibn abi talib", "ibn abi talib", "abu turab",
            "haidar", "haider", "murtaza", "murtada", "fourth caliph",
        ),
        description=(
            "Life and caliphate of Ali ibn Abi Talib (35-40 AH): lineage, early life "
            "in the Prophet's household, first to accept Islam, marriage to Fatimah, "
            "battles and the flag at Khaybar, his scholarship, the Sunni approach to "
            "the civil wars, the pledge of Talha and Zubayr, Muawiyah's refusal and "
            "the demand for revenge for Uthman, the Battle of the Camel (Jamal) and "
            "Aisha, the Battle of Siffin, death of Ammar ibn Yasir, Qurans on spears, "
            "the arbitration, the birth of the Khawarij, Ibn Abbas's debate at Harura, "
            "the Battle of Nahrawan, his assassination in 40 AH, final advice to his "
            "sons, and the Year of Union."
        ),
    ),
)

DOCS_BY_ID: dict[str, SourceDoc] = {d.doc_id: d for d in DOCUMENTS}
DOCS_BY_FILE: dict[str, SourceDoc] = {d.file: d for d in DOCUMENTS}
DOC_IDS: tuple[str, ...] = tuple(DOCS_BY_ID)


def normalize(text: str) -> str:
    """Lowercase, strip diacritics and punctuation, collapse whitespace.

    "Abū-Bakr's" -> "abu bakr s", so aliases match regardless of transliteration
    marks, hyphens or apostrophes.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return f" {text.strip()} "


_ALIAS_PATTERNS: dict[str, tuple[str, ...]] = {
    d.doc_id: tuple(normalize(a) for a in d.aliases) for d in DOCUMENTS
}


def match_aliases(question: str) -> list[str]:
    """Return doc_ids whose aliases appear as whole words in the question."""
    q = normalize(question)
    return [
        doc_id
        for doc_id, aliases in _ALIAS_PATTERNS.items()
        if any(alias in q for alias in aliases)
    ]


def catalog_prompt() -> str:
    """Catalog rendered for the LLM router."""
    return "\n".join(f"- {d.doc_id}: {d.description}" for d in DOCUMENTS)
