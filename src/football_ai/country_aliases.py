"""country_aliases.py
===================
Maps alternative / official country name spellings to a single canonical
**common-English** name so that team lookups are consistent across files
from different data providers (FIFA, Wikipedia, Opta, ESPN, etc.).

Usage::

    from .country_aliases import resolve_country

    canonical = resolve_country("Korea Republic")   # → "South Korea"
    canonical = resolve_country("IR Iran")          # → "Iran"
    canonical = resolve_country("côte d'ivoire")    # → "Ivory Coast"
"""

from __future__ import annotations

import unicodedata


# ---------------------------------------------------------------------------
# Alias → Canonical dictionary
# Key  : lower-cased, accent-stripped variant
# Value: canonical common-English name (title-cased)
# ---------------------------------------------------------------------------
COUNTRY_ALIASES: dict[str, str] = {
    # ---- Afghanistan ----
    "afghanistan": "Afghanistan",

    # ---- Albania ----
    "albania": "Albania",
    "shqiperia": "Albania",

    # ---- Algeria ----
    "algeria": "Algeria",
    "algerie": "Algeria",
    "al jazair": "Algeria",

    # ---- Andorra ----
    "andorra": "Andorra",

    # ---- Angola ----
    "angola": "Angola",

    # ---- Argentina ----
    "argentina": "Argentina",
    "arg": "Argentina",

    # ---- Armenia ----
    "armenia": "Armenia",

    # ---- Australia ----
    "australia": "Australia",
    "socceroos": "Australia",
    "aus": "Australia",

    # ---- Austria ----
    "austria": "Austria",
    "osterreich": "Austria",

    # ---- Azerbaijan ----
    "azerbaijan": "Azerbaijan",

    # ---- Bahrain ----
    "bahrain": "Bahrain",

    # ---- Bangladesh ----
    "bangladesh": "Bangladesh",

    # ---- Belarus ----
    "belarus": "Belarus",
    "byelorussia": "Belarus",

    # ---- Belgium ----
    "belgium": "Belgium",
    "belgique": "Belgium",
    "belgie": "Belgium",
    "bel": "Belgium",

    # ---- Bolivia ----
    "bolivia": "Bolivia",
    "plurinational state of bolivia": "Bolivia",

    # ---- Bosnia and Herzegovina ----
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina",
    "bih": "Bosnia and Herzegovina",
    "bih men's national football team": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",

    # ---- Brazil ----
    "brazil": "Brazil",
    "brasil": "Brazil",
    "bra": "Brazil",

    # ---- Bulgaria ----
    "bulgaria": "Bulgaria",

    # ---- Burkina Faso ----
    "burkina faso": "Burkina Faso",

    # ---- Cameroon ----
    "cameroon": "Cameroon",
    "cameroun": "Cameroon",

    # ---- Canada ----
    "canada": "Canada",
    "can": "Canada",

    # ---- Chile ----
    "chile": "Chile",
    "chi": "Chile",

    # ---- China ----
    "china": "China",
    "china pr": "China",
    "peoples republic of china": "China",
    "people's republic of china": "China",
    "prc": "China",
    "chn": "China",

    # ---- Colombia ----
    "colombia": "Colombia",
    "col": "Colombia",

    # ---- Congo ----
    "congo": "Congo",
    "republic of the congo": "Congo",
    "congo-brazzaville": "Congo",

    # ---- DR Congo ----
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo dr": "DR Congo",
    "drc": "DR Congo",
    "zaire": "DR Congo",

    # ---- Costa Rica ----
    "costa rica": "Costa Rica",
    "crc": "Costa Rica",

    # ---- Croatia ----
    "croatia": "Croatia",
    "hrvatska": "Croatia",
    "cro": "Croatia",

    # ---- Cuba ----
    "cuba": "Cuba",

    # ---- Cyprus ----
    "cyprus": "Cyprus",

    # ---- Czech Republic ----
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "czechoslovakia": "Czech Republic",
    "cze": "Czech Republic",

    # ---- Denmark ----
    "denmark": "Denmark",
    "den": "Denmark",

    # ---- Ecuador ----
    "ecuador": "Ecuador",
    "ecu": "Ecuador",

    # ---- Egypt ----
    "egypt": "Egypt",
    "arab republic of egypt": "Egypt",

    # ---- El Salvador ----
    "el salvador": "El Salvador",
    "salv": "El Salvador",

    # ---- England ----
    "england": "England",
    "eng": "England",
    "three lions": "England",

    # ---- Equatorial Guinea ----
    "equatorial guinea": "Equatorial Guinea",
    "guinea ecuatorial": "Equatorial Guinea",

    # ---- Estonia ----
    "estonia": "Estonia",

    # ---- Ethiopia ----
    "ethiopia": "Ethiopia",

    # ---- Faroe Islands ----
    "faroe islands": "Faroe Islands",
    "faroes": "Faroe Islands",
    "fro": "Faroe Islands",

    # ---- Finland ----
    "finland": "Finland",
    "fin": "Finland",

    # ---- France ----
    "france": "France",
    "fra": "France",

    # ---- Gabon ----
    "gabon": "Gabon",

    # ---- Georgia ----
    "georgia": "Georgia",
    "geo": "Georgia",

    # ---- Germany ----
    "germany": "Germany",
    "deutschland": "Germany",
    "west germany": "Germany",
    "ger": "Germany",

    # ---- Ghana ----
    "ghana": "Ghana",
    "gha": "Ghana",
    "black stars": "Ghana",

    # ---- Greece ----
    "greece": "Greece",
    "hellas": "Greece",
    "ellada": "Greece",

    # ---- Guatemala ----
    "guatemala": "Guatemala",

    # ---- Guinea ----
    "guinea": "Guinea",

    # ---- Haiti ----
    "haiti": "Haiti",

    # ---- Honduras ----
    "honduras": "Honduras",
    "hon": "Honduras",

    # ---- Hungary ----
    "hungary": "Hungary",
    "magyarorszag": "Hungary",

    # ---- Iceland ----
    "iceland": "Iceland",
    "isl": "Iceland",

    # ---- India ----
    "india": "India",
    "ind": "India",

    # ---- Indonesia ----
    "indonesia": "Indonesia",

    # ---- Iran ----
    "iran": "Iran",
    "ir iran": "Iran",
    "islamic republic of iran": "Iran",
    "persia": "Iran",
    "irn": "Iran",

    # ---- Iraq ----
    "iraq": "Iraq",
    "irq": "Iraq",

    # ---- Ireland ----
    "ireland": "Ireland",
    "republic of ireland": "Ireland",
    "irl": "Ireland",

    # ---- Israel ----
    "israel": "Israel",
    "isr": "Israel",

    # ---- Italy ----
    "italy": "Italy",
    "italia": "Italy",
    "ita": "Italy",
    "azzurri": "Italy",

    # ---- Ivory Coast ----
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "la cote d'ivoire": "Ivory Coast",
    "civ": "Ivory Coast",

    # ---- Jamaica ----
    "jamaica": "Jamaica",
    "jam": "Jamaica",

    # ---- Japan ----
    "japan": "Japan",
    "jpn": "Japan",
    "samurai blue": "Japan",

    # ---- Jordan ----
    "jordan": "Jordan",
    "jor": "Jordan",

    # ---- Kazakhstan ----
    "kazakhstan": "Kazakhstan",

    # ---- Kenya ----
    "kenya": "Kenya",

    # ---- Kosovo ----
    "kosovo": "Kosovo",
    "republic of kosovo": "Kosovo",

    # ---- Kuwait ----
    "kuwait": "Kuwait",

    # ---- Latvia ----
    "latvia": "Latvia",

    # ---- Lebanon ----
    "lebanon": "Lebanon",
    "liban": "Lebanon",

    # ---- Libya ----
    "libya": "Libya",

    # ---- Liechtenstein ----
    "liechtenstein": "Liechtenstein",

    # ---- Lithuania ----
    "lithuania": "Lithuania",

    # ---- Luxembourg ----
    "luxembourg": "Luxembourg",

    # ---- Madagascar ----
    "madagascar": "Madagascar",

    # ---- Mali ----
    "mali": "Mali",

    # ---- Malta ----
    "malta": "Malta",

    # ---- Mexico ----
    "mexico": "Mexico",
    "mejico": "Mexico",
    "mex": "Mexico",
    "el tri": "Mexico",

    # ---- Moldova ----
    "moldova": "Moldova",
    "republic of moldova": "Moldova",

    # ---- Montenegro ----
    "montenegro": "Montenegro",
    "mne": "Montenegro",

    # ---- Morocco ----
    "morocco": "Morocco",
    "maroc": "Morocco",
    "al maghrib": "Morocco",
    "mar": "Morocco",

    # ---- Mozambique ----
    "mozambique": "Mozambique",

    # ---- Netherlands ----
    "netherlands": "Netherlands",
    "holland": "Netherlands",
    "ned": "Netherlands",
    "dutch": "Netherlands",

    # ---- New Zealand ----
    "new zealand": "New Zealand",
    "nzl": "New Zealand",
    "all whites": "New Zealand",

    # ---- Nigeria ----
    "nigeria": "Nigeria",
    "nga": "Nigeria",
    "super eagles": "Nigeria",

    # ---- Northern Ireland ----
    "northern ireland": "Northern Ireland",
    "nir": "Northern Ireland",

    # ---- North Korea ----
    "north korea": "North Korea",
    "dpr korea": "North Korea",
    "democratic people's republic of korea": "North Korea",
    "korea dpr": "North Korea",
    "prk": "North Korea",

    # ---- North Macedonia ----
    "north macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "republic of north macedonia": "North Macedonia",
    "former yugoslav republic of macedonia": "North Macedonia",
    "fyrom": "North Macedonia",
    "mkd": "North Macedonia",

    # ---- Norway ----
    "norway": "Norway",
    "norge": "Norway",
    "nor": "Norway",

    # ---- Oman ----
    "oman": "Oman",
    "oman sultanate": "Oman",

    # ---- Pakistan ----
    "pakistan": "Pakistan",

    # ---- Palestine ----
    "palestine": "Palestine",
    "state of palestine": "Palestine",

    # ---- Panama ----
    "panama": "Panama",
    "pan": "Panama",

    # ---- Paraguay ----
    "paraguay": "Paraguay",
    "par": "Paraguay",

    # ---- Peru ----
    "peru": "Peru",
    "per": "Peru",

    # ---- Philippines ----
    "philippines": "Philippines",
    "pilipinas": "Philippines",

    # ---- Poland ----
    "poland": "Poland",
    "polska": "Poland",
    "pol": "Poland",

    # ---- Portugal ----
    "portugal": "Portugal",
    "por": "Portugal",

    # ---- Qatar ----
    "qatar": "Qatar",
    "qat": "Qatar",

    # ---- Romania ----
    "romania": "Romania",
    "rou": "Romania",

    # ---- Russia ----
    "russia": "Russia",
    "russian federation": "Russia",
    "soviet union": "Russia",
    "ussr": "Russia",
    "rus": "Russia",

    # ---- Saudi Arabia ----
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "ksa (saudi arabia)": "Saudi Arabia",
    "ksa men's national football team": "Saudi Arabia",
    "ksa national football team": "Saudi Arabia",
    "ksa football team": "Saudi Arabia",
    "saudi": "Saudi Arabia",

    # ---- Scotland ----
    "scotland": "Scotland",
    "sco": "Scotland",

    # ---- Senegal ----
    "senegal": "Senegal",
    "sen": "Senegal",

    # ---- Serbia ----
    "serbia": "Serbia",
    "yugoslavia": "Serbia",
    "srb": "Serbia",

    # ---- Sierra Leone ----
    "sierra leone": "Sierra Leone",

    # ---- Slovakia ----
    "slovakia": "Slovakia",
    "svk": "Slovakia",

    # ---- Slovenia ----
    "slovenia": "Slovenia",
    "svn": "Slovenia",

    # ---- Somalia ----
    "somalia": "Somalia",

    # ---- South Africa ----
    "south africa": "South Africa",
    "rsa": "South Africa",
    "bafana bafana": "South Africa",

    # ---- South Korea ----
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "korea (south)": "South Korea",
    "kor": "South Korea",

    # ---- Spain ----
    "spain": "Spain",
    "espana": "Spain",
    "esp": "Spain",

    # ---- Sweden ----
    "sweden": "Sweden",
    "sverige": "Sweden",
    "swe": "Sweden",

    # ---- Switzerland ----
    "switzerland": "Switzerland",
    "schweiz": "Switzerland",
    "suisse": "Switzerland",
    "svizzera": "Switzerland",
    "sui": "Switzerland",

    # ---- Syria ----
    "syria": "Syria",
    "syr": "Syria",

    # ---- Tanzania ----
    "tanzania": "Tanzania",

    # ---- Thailand ----
    "thailand": "Thailand",
    "tha": "Thailand",

    # ---- Togo ----
    "togo": "Togo",
    "tog": "Togo",

    # ---- Trinidad and Tobago ----
    "trinidad and tobago": "Trinidad and Tobago",
    "trinidad & tobago": "Trinidad and Tobago",
    "tnt": "Trinidad and Tobago",
    "tri": "Trinidad and Tobago",

    # ---- Tunisia ----
    "tunisia": "Tunisia",
    "tun": "Tunisia",

    # ---- Turkey ----
    "turkey": "Turkey",
    "turkiye": "Turkey",
    "tur": "Turkey",

    # ---- Uganda ----
    "uganda": "Uganda",

    # ---- Ukraine ----
    "ukraine": "Ukraine",
    "ukr": "Ukraine",

    # ---- United Arab Emirates ----
    "united arab emirates": "United Arab Emirates",
    "uae": "United Arab Emirates",

    # ---- United States ----
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "united states of america": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "usmnt": "United States",

    # ---- Uruguay ----
    "uruguay": "Uruguay",
    "uru": "Uruguay",

    # ---- Uzbekistan ----
    "uzbekistan": "Uzbekistan",
    "uzb": "Uzbekistan",

    # ---- Venezuela ----
    "venezuela": "Venezuela",
    "ven": "Venezuela",

    # ---- Vietnam ----
    "vietnam": "Vietnam",
    "viet nam": "Vietnam",
    "vnm": "Vietnam",

    # ---- Wales ----
    "wales": "Wales",
    "cymru": "Wales",
    "wal": "Wales",

    # ---- Zambia ----
    "zambia": "Zambia",
    "zam": "Zambia",

    # ---- Zimbabwe ----
    "zimbabwe": "Zimbabwe",
    "zim": "Zimbabwe",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics/accents from a string using NFKD normalisation."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def resolve_country(name: str) -> str:
    """Return the canonical common-English country name for *name*.

    Steps:
    1. Strip leading/trailing whitespace.
    2. NFKD-normalise to remove accents (so ``"Côte d'Ivoire"`` → ``"cote d'ivoire"``).
    3. Lower-case.
    4. Look up in ``COUNTRY_ALIASES``; return canonical if found.
    5. Otherwise return the original string title-cased (preserving unknown names).

    Parameters
    ----------
    name:
        Raw country/team name from any data source.

    Returns
    -------
    str
        Canonical common-English name, or title-cased original if unknown.
    """
    if not isinstance(name, str):
        return str(name).strip()
    name = name.strip()
    if not name:
        return name
    key = _strip_accents(name).lower()
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    # Try after collapsing internal whitespace
    key_collapsed = " ".join(key.split())
    if key_collapsed in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key_collapsed]
    # Unknown — return title-cased original (preserves club names, etc.)
    return " ".join(name.split()).title()
