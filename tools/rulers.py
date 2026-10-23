"""Heads of state on 1 January 1700, one per AoE country (build_history.write_characters).

Each entry is ruler(first name, last name, birth date, traits, **overrides). The
birth date is "Y.M.D", a year when only that is known, or None. Traits are a Vic3
personality, sometimes followed by a skill or condition. Overrides: culture
(when the ruler's isn't the country's primary culture), female, interest_group,
ideology. Without overrides the interest group and ideology come from the
country's government (RULER_POLITICS in build_history.py).

generic(traits) is a ruler the game names from the country's culture: rotating
offices (city councils, Lucca's gonfaloniere, Ragusa's rector) and rulers the
sources don't name.

Personal-union juniors (SCO, SAX, PRU) have no entry: a Vic3 personal union
shows the overlord's ruler. Same-ruler pairs outside the mod's unions get a
local head instead (NET's grand pensionary, ARN's viceroy, SIE's governor),
except KOL/LIG, where Joseph Clemens appears twice.
"""


def ruler(first, last, born, traits, **extra):
    return dict(first=first, last=last, born=born, traits=traits.split(), **extra)


def generic(traits, **extra):
    return dict(first=None, last=None, born=None, traits=traits.split(), **extra)


RULERS = {
    # ------------------------------------------------------------ Europe ----
    "AUS": ruler("Leopold I", "von Habsburg", "1640.6.9", "pious"),
    "FRA": ruler("Louis XIV", "de Bourbon", "1638.9.5", "imperious", ideology="ideology_legitimist"),
    "GBR": ruler("William III", "of Orange", "1650.11.14", "reserved", culture="dutch"),
    "NET": ruler("Anthonie", "Heinsius", "1641.11.23", "tactful basic_diplomat"),   # grand pensionary
    "SPA": ruler("Carlos II", "de Austria", "1661.11.6", "compliant sickly"),
    "ARN": ruler("Georg", "von Hessen-Darmstadt", "1669.4.25", "brave", culture="north_german",
                 interest_group="ig_armed_forces", ideology="ideology_royalist"),   # viceroy of Catalonia
    "POR": ruler("Pedro II", "de Bragança", "1648.4.26", "cautious"),
    "DNN": ruler("Frederik IV", "af Oldenburg", "1671.10.11", "romantic"),
    "HOL": ruler("Friedrich IV", "von Holstein-Gottorp", "1671.10.18", "brave"),
    "SWE": ruler("Karl XII", "av Pfalz-Zweibrücken", "1682.6.17", "brave"),
    "RUS": ruler("Pyotr I", "Romanov", "1672.6.9", "innovative"),
    "PLC": ruler("August II", "Wettin", "1670.5.12", "hedonist", culture="north_german"),   # and SAX
    "BRN": ruler("Friedrich III", "von Hohenzollern", "1657.7.11", "ambitious"),          # and PRU
    "KUR": ruler("Friedrich Wilhelm", "Kettler", "1692.7.19", "compliant"),   # a minor, under his uncle's regency
    "TUR": ruler("Mustafa II", "Osmanoğlu", "1664.2.6", "brave"),
    "CRI": ruler("Devlet II", "Giray", 1648, "ambitious"),
    "WAL": ruler("Constantin", "Brâncoveanu", "1654.8.15", "aesthete"),
    "MOL": ruler("Antioh", "Cantemir", 1670, "cautious"),
    "SIE": ruler("György", "Bánffy", 1661, "cautious"),   # governor of Transylvania
    "HET": ruler("Ivan", "Mazepa", "1639.3.30", "ambitious"),
    "ZZH": ruler("Petro", "Pryma", None, "direct", interest_group="ig_armed_forces",
                 ideology="ideology_traditionalist"),   # kish otaman
    "DON": ruler("Frol", "Minaev", None, "cautious"),     # host ataman
    "TRK": generic("brave"),
    "VEN": ruler("Silvestro", "Valier", "1630.3.28", "tactful"),
    "GEN": ruler("Girolamo", "De Mari", 1644, "meticulous"),
    "LUC": generic("cautious"),
    "DLM": generic("tactful"),
    "PAP": ruler("Innocent XII", "Pignatelli", "1615.3.13", "honorable"),
    "SVY": ruler("Vittorio Amedeo II", "di Savoia", "1666.5.14", "ambitious"),
    "TUS": ruler("Cosimo III", "de' Medici", "1642.8.14", "bigoted"),
    "MOD": ruler("Rinaldo", "d'Este", "1655.4.26", "cautious"),
    "PAR": ruler("Francesco", "Farnese", "1678.5.19", "reserved"),
    "MNT": ruler("Ferdinando Carlo", "Gonzaga", "1652.8.31", "hedonist"),
    "MLT": ruler("Ramon", "Perellos y Roccaful", 1637, "pious", culture="catalan"),
    "SWI": ruler("Heinrich", "Escher", 1626, "tactful"),   # burgomaster of Zürich, the Vorort
    "LOR": ruler("Léopold", "de Lorraine", "1679.9.11", "honorable", culture="french"),
    "BAV": ruler("Maximilian II Emanuel", "von Wittelsbach", "1662.7.11", "ambitious"),
    "PAL": ruler("Johann Wilhelm", "von Pfalz-Neuburg", "1658.4.19", "aesthete"),
    "KOL": ruler("Joseph Clemens", "von Wittelsbach", "1671.12.5", "hedonist", culture="south_german"),
    "LIG": ruler("Joseph Clemens", "von Wittelsbach", "1671.12.5", "hedonist", culture="south_german"),
    "MUN": ruler("Friedrich Christian", "von Plettenberg", "1644.8.8", "meticulous"),
    "MNZ": ruler("Lothar Franz", "von Schönborn", "1655.10.4", "aesthete", culture="south_german"),
    "WUL": ruler("Johann Philipp", "von Greiffenclau", "1652.2.13", "cautious"),
    "TRR": ruler("Johann Hugo", "von Orsbeck", "1634.1.13", "pious"),
    "SAL": ruler("Johann Ernst", "von Thun", "1643.7.3", "pious"),
    "FUL": ruler("Placidus", "von Droste", 1641, "pious"),
    "AUG": ruler("Alexander Sigismund", "von Pfalz-Neuburg", "1663.4.16", "reserved",
                 interest_group="ig_devout", ideology="ideology_theocrat"),   # prince-bishop
    "BAD": ruler("Ludwig Wilhelm", "von Baden-Baden", "1655.4.8", "brave"),
    "ANS": ruler("Georg Friedrich II", "von Brandenburg-Ansbach", "1678.5.3", "reckless"),
    "ANH": ruler("Leopold I", "von Anhalt-Dessau", "1676.7.3", "direct"),
    "EFR": ruler("Christian Eberhard", "Cirksena", "1665.10.1", "cautious"),
    "MEC": ruler("Friedrich Wilhelm", "von Mecklenburg", "1675.3.28", "persistent"),
    "WUR": ruler("Eberhard Ludwig", "von Württemberg", "1676.9.18", "hedonist"),
    "NAS": ruler("Johann Ernst", "von Nassau-Weilburg", "1664.6.13", "direct"),
    "BRA": ruler("Rudolf August", "von Braunschweig-Wolfenbüttel", "1627.5.16", "compliant"),
    "HAN": ruler("Georg Ludwig", "von Hannover", "1660.6.7", "reserved"),
    "HES": ruler("Ernst Ludwig", "von Hessen-Darmstadt", "1667.12.15", "hedonist"),
    "LAU": ruler("Georg Wilhelm", "von Braunschweig-Lüneburg", "1624.1.26", "romantic"),
    "COB": ruler("Friedrich II", "von Sachsen-Gotha-Altenburg", "1676.7.28", "ambitious"),
    "MEI": ruler("Bernhard I", "von Sachsen-Meiningen", "1649.9.10", "pious"),
    "WEI": ruler("Wilhelm Ernst", "von Sachsen-Weimar", "1662.10.19", "imperious"),
    "LIP": ruler("Friedrich Adolf", "zur Lippe", "1667.9.2", "cautious"),
    "SAA": ruler("Ludwig Crato", "von Nassau-Saarbrücken", "1663.3.28", "tactful"),
    "SGM": ruler("Meinrad II", "von Hohenzollern-Sigmaringen", "1673.11.1", "honorable"),
    "TKL": ruler("Johann Adolf", "von Bentheim-Tecklenburg", "1637.9.22", "persistent"),
    "HAM": generic("meticulous"),
    "LUB": generic("cautious"),
    "FRM": generic("tactful"),
    "ACH": generic("reserved"),
    "DOR": generic("persistent"),
    "ULM": generic("meticulous"),
    "NUR": generic("cautious"),

    # --------------------------------------------- colonies and viceroys ----
    "EIC": ruler("Charles", "Eyre", None, "meticulous", interest_group="ig_industrialists",
                 ideology="ideology_moderate"),   # president of Fort William
    "THC": ruler("Richard", "Coote", 1636, "honorable", culture="british", interest_group="ig_landowners",
                 ideology="ideology_moderate"),   # Earl of Bellomont
    "VNS": ruler("José", "Sarmiento de Valladares", 1643, "cautious", culture="spanish",
                 interest_group="ig_landowners", ideology="ideology_moderate"),
    "VOP": ruler("Melchor", "Portocarrero", 1636, "brave", culture="spanish", interest_group="ig_landowners",
                 ideology="ideology_moderate"),   # Count of Monclova

    # ------------------------------------------- Mughal India and subahs ----
    "MUG": ruler("Aurangzeb", "Timurid", "1618.11.3", "pious", interest_group="ig_devout",
                 ideology="ideology_theocrat"),
    "AFG": ruler("Muazzam", "Timurid", "1643.10.14", "cautious"),       # Kabul
    "BNG": ruler("Azim-ush-Shan", "Timurid", "1664.12.15", "ambitious"),
    "BIH": ruler("Fidai", "Khan", None, "direct"),
    "PNJ": ruler("Abu Nasr", "Khan", None, "cautious"),                 # Lahore
    "KAS": ruler("Fazil", "Khan", None, "meticulous"),
    "SIN": ruler("Hifzullah", "Khan", None, "persistent"),              # Thatta
    "MAW": ruler("Mukhtiyar", "Khan", None, "cautious"),
    "GUJ": ruler("Shujaat", "Khan", None, "persistent"),
    "KRN": ruler("Zulfiqar", "Khan", 1657, "ruthless"),
    "GOL": ruler("Jan Sipar", "Khan", None, "honorable"),               # Hyderabad
    "BRR": ruler("Ghazi ud-Din", "Khan", None, "brave"),
    "RAJ": ruler("Swarup Singh", "Rathore", 1689, "compliant", interest_group="ig_landowners",
                 ideology="ideology_traditionalist"),                   # Bikaner
    "MUL": generic("cautious"),
    "AWA": generic("reserved"),
    "ORI": generic("cautious"),
    "BIJ": generic("direct"),
    "DLT": generic("persistent"),

    # ----------------------------------------------------- rest of India ----
    "MAH": ruler("Rajaram", "Bhonsle", "1670.2.24", "persistent"),
    "SKH": ruler("Gobind Singh", "Sodhi", "1666.12.22", "brave"),
    "JAI": ruler("Jai Singh II", "Kachhwaha", "1688.11.3", "innovative"),
    "JOD": ruler("Ajit Singh", "Rathore", "1679.2.19", "ambitious"),
    "MEW": ruler("Amar Singh II", "Sisodia", "1672.10.3", "tactful"),
    "BUN": ruler("Chhatrasal", "Bundela", "1649.5.4", "brave"),
    "REW": ruler("Avadhut Singh", "Baghel", None, "compliant"),
    "DGH": ruler("Bakht Buland Shah", "Gond", None, "ambitious"),
    "GRH": ruler("Narendra Shah", "Gond", None, "cautious"),
    "BAS": ruler("Digpal Deo", "Kakatiya", None, "reserved"),
    "JEY": ruler("Raghunath Krishna Deo", "Suryavanshi", None, "cautious"),
    "RTP": ruler("Rajsingh Deo", "Haihaya", None, "reserved"),
    "SBP": ruler("Chhatra Singh", "Chauhan", None, "direct"),
    "SGJ": ruler("Baiha Dadu", "Raksel", None, "reserved"),
    "SHI": generic("cautious"),
    "CRL": generic("brave"),
    "ASM": ruler("Rudra Singha", "Ahom", None, "ambitious"),
    "DMS": ruler("Tamradhwaj Narayan", "Dimasa", None, "brave"),
    "MNP": ruler("Charairongba", "Ningthouja", None, "pious"),
    "TRP": ruler("Ratna Manikya II", "Manikya", None, "compliant"),
    "BHU": ruler("Gedun", "Chomphel", None, "reserved", interest_group="ig_devout",
                 ideology="ideology_theocrat"),                         # Druk Desi
    "SIK": ruler("Chakdor", "Namgyal", 1686, "innovative"),
    "LAD": ruler("Nyima", "Namgyal", None, "pious"),
    "NEP": ruler("Bhaskara", "Malla", None, "reckless"),
    "MYS": ruler("Chikka Devaraja", "Wodeyar", "1645.9.22", "meticulous"),
    "KLD": ruler("Basavappa Nayaka I", "Keladi", None, "pious"),
    "CHG": ruler("Bharamanna Nayaka", "Chitradurga", None, "ambitious"),
    "MDR": ruler("Mangammal", "Nayak", None, "tactful", female=True),   # regent for her grandson
    "TRA": ruler("Ravi Varma", "Venad", None, "cautious"),
    "COC": ruler("Rama Varma", "Perumpadappu", None, "reserved"),
    "CIT": ruler("Manavikraman", "Nediyiruppu", None, "tactful"),       # the Zamorin
    "KND": ruler("Vimaladharmasuriya II", "Dinajara", None, "pious"),
    "MLD": ruler("Muhammad Muhyiddin", "Isdhoo", None, "pious"),

    # ------------------------------------- Persia, Arabia, Central Asia ----
    "PER": ruler("Soltan Hoseyn", "Safavi", 1668, "pious"),
    "OMA": ruler("Saif I", "al-Ya'rubi", None, "ambitious"),
    "IMO": generic("pious"),
    "NYE": ruler("al-Mahdi Muhammad", "al-Qasimi", 1638, "imperious"),  # Zaydi imam
    "NJN": ruler("Hibat Allah", "al-Makrami", None, "pious"),
    "KTH": ruler("Ja'far ibn Badr", "Al Kathiri", None, "cautious"),
    "MHR": ruler("Abdallah", "Afrar", None, "direct"),
    "HSA": ruler("Sa'dun", "Al Humaid", None, "ambitious"),
    "NEJ": generic("pious"),
    "ABU": generic("cautious"),
    "QSM": generic("brave"),
    "GZK": ruler("Surkhay", "Shamkhal", None, "wrathful"),
    "BUK": ruler("Subhan Quli", "Ashtarkhanid", None, "pious"),
    "KHI": ruler("Shah Niyaz", "Arabshahid", None, "direct"),
    "KAZ": ruler("Tauke", "Töre", None, "meticulous"),
    "KAL": ruler("Samandar", "Ahmadzai", None, "brave"),
    "MAK": generic("reserved"),
    "BDK": generic("cautious"),
    "DZN": ruler("Tsewang Rabtan", "Choros", 1663, "ambitious"),
    "KMK": ruler("Ayuka", "Torghut", 1642, "tactful"),
    "TIB": ruler("Lha-bzang", "Khoshut", None, "ambitious", culture="oirat"),
    "MGL": ruler("Zanabazar", "Borjigin", 1635, "pious", interest_group="ig_devout",
                 ideology="ideology_theocrat"),
    "KML": ruler("Abdullah", "Beg", None, "cautious"),
    "XIN": ruler("Muhammad Mu'min", "Chagatayid", None, "compliant"),   # Yarkent

    # ---------------------------------------------------------- East Asia ----
    "QNG": ruler("Kangxi", "Aisin Gioro", "1654.5.4", "meticulous"),
    "KOR": ruler("Sukjong", "Yi", "1661.10.7", "imperious"),
    "TKG": ruler("Tsunayoshi", "Tokugawa", "1646.2.23", "pious"),
    "SAT": ruler("Tsunataka", "Shimazu", 1675, "meticulous"),
    "SEN": ruler("Tsunamura", "Date", "1659.4.29", "tactful"),
    "YZW": ruler("Tsunanori", "Uesugi", "1663.11.27", "hedonist"),
    "TOS": ruler("Toyomasa", "Yamauchi", 1641, "cautious"),

    # ---------------------------------------------------- Southeast Asia ----
    "NGU": ruler("Phúc Chu", "Nguyễn", "1675.6.11", "pious"),
    "TRH": ruler("Căn", "Trịnh", 1633, "cautious"),
    "AYO": ruler("Phetracha", "Ban Phlu Luang", 1632, "ruthless"),
    "BUR": ruler("Sanay", "Taungoo", 1673, "cautious"),
    "ARK": ruler("Naradipati", "Mrauk-U", None, "reckless"),
    "CMG": generic("reserved"),
    "LXA": ruler("Setthathirath II", "Khun Lo", None, "pious"),
    "CAM": ruler("Chey Chettha IV", "Varman", None, "persistent"),
    "CHP": ruler("Po Saktiraydaputih", "Panduranga", None, "tactful"),
    "JAV": ruler("Amangkurat II", "Mataram", None, "compliant"),
    "BBG": ruler("Pangeran Putra", "Blambangan", None, "brave"),
    "BAL": ruler("Dewa Agung Jambe I", "Kepakisan", None, "ambitious"),
    "ATJ": ruler("Badr ul-Alam", "Jamal ul-Lail", None, "pious"),
    "JOH": ruler("Abdul Jalil IV", "Bendahara", None, "cautious"),
    "PAT": ruler("Raja Emas", "Kelantan", None, "reserved"),
    "KED": ruler("Abdullah Muadzam Shah", "Kedah", None, "cautious"),
    "PRK": ruler("Mahmud Iskandar Shah", "Perak", None, "persistent"),
    "PYG": ruler("Indermasyah", "Pagaruyung", None, "reserved"),
    "PLM": ruler("Abdurrahman", "Palembang", None, "pious"),
    "KLM": ruler("Tahmidullah I", "Banjar", None, "ambitious"),
    "BRU": ruler("Nasruddin", "Bolkiah", None, "cautious"),
    "SLW": ruler("La Patau", "Matanna Tikka", 1672, "tactful"),          # Bone
    "MAL": ruler("Said Fatahullah", "Ternate", None, "compliant"),
    "MGW": ruler("Kahar ud-Din Kuda", "Maguindanao", None, "brave"),
    "SUL": ruler("Shahab ud-Din", "Sulu", None, "pious"),
    "DJA": generic("cautious"),
    "BRS": generic("reserved"),
    "PTK": generic("ambitious"),
    "BTN": generic("pious"),
    "KTI": generic("cautious"),
    "HAW": ruler("Keawe'ikekahiali'iokamoku", "Hawai'i", None, "ambitious"),

    # ------------------------------------------------------------ Africa ----
    "MOR": ruler("Ismail", "Alaouite", 1645, "cruel"),
    "ALD": ruler("Hajji Mustafa", "Dey", None, "brave"),
    "TRZ": ruler("Addi I", "Awlad Daman", None, "brave"),
    "ETH": ruler("Iyasu I", "Solomonid", 1654, "honorable"),
    "HAR": ruler("Talha", "ibn Abbas", None, "pious"),
    "SUD": ruler("Badi III", "Unsab", None, "ambitious"),
    "ASH": ruler("Osei Tutu", "Oyoko", 1660, "ambitious"),
    "DAH": ruler("Akaba", "Alladaxonou", None, "brave"),
    "BEN": ruler("Ewuakpe", "Eweka", None, "reserved"),
    "OYO": ruler("Osiyago", "Oranyan", None, "imperious"),
    "KON": ruler("Pedro IV", "Água Rosada", None, "tactful"),
    "MAD": ruler("Andriamasinavalona", "Merina", None, "ambitious"),
    "AWS": generic("pious"),
    "GEL": generic("cautious"),
    "MAJ": generic("brave"),
    "WRL": generic("direct"),
    "ARO": generic("tactful"),
    "CLA": generic("ambitious"),
    "WRI": generic("cautious"),
    "GBU": generic("brave"),
    "GFL": generic("pious"),
    "SEG": generic("ambitious"),
    "ZAN": generic("cautious"),
    "SFL": generic("reserved"),
    "GAZ": generic("brave"),
    "AGC": generic("cautious"),

    # ---------------------------------------------------------- Americas ----
    "MSK": ruler("Jeremy I", "Miskito", None, "tactful"),
    "IQS": generic("tactful"),
    "ABB": generic("brave"),
    "APC": generic("brave"),
    "CMN": generic("reckless"),
    "LNP": generic("cautious"),
}
