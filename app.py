import io
import re
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process


st.set_page_config(
    page_title="Auditor de EPIs — PGR x Sistema",
    page_icon="🦺",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Normalização
# -----------------------------------------------------------------------------

ROMAN_WORDS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}
FILLER_WORDS = {"DE", "DA", "DO", "DAS", "DOS", "EM", "E"}

TOKEN_REPLACEMENTS = {
    "SEG": "SEGURANCA",
    "SEGUR": "SEGURANCA",
    "TEC": "TECNICO",
    "TECN": "TECNICO",
    "AUX": "AUXILIAR",
    "ENG": "ENGENHEIRO",
    "OP": "OPERADOR",
    "OPER": "OPERADOR",
    "SUP": "SUPERVISOR",
    "COORD": "COORDENADOR",
    "ADM": "ADMINISTRATIVO",
    "RESP": "RESPIRADOR",
    "RESPIR": "RESPIRADOR",
    "MOV": "MOVIMENTACAO",
    "MEC": "MECANICO",
    "SOLDAR": "SOLDA",
    "SOLDADURA": "SOLDA",
    # Equivalência de nomenclatura usada entre PGR e sistema.
    # Ex.: "Técnico Orçamentista" <-> "Técnico de Orçamento".
    "ORCAMENTISTA": "ORCAMENTO",
    "ANDAIMES": "ANDAIME",
    "CARGAS": "CARGA",
    "OBRAS": "OBRA",
    "MATERIAIS": "MATERIAL",
    "SERVICOS": "SERVICO",
}


def strip_accents(text: str) -> str:
    text = "" if text is None else str(text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, keep_numbers: bool = True) -> str:
    """Normalização agressiva para comparação, sem perder números importantes."""
    text = strip_accents(text).upper()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    text = re.sub(r"[()\[\]{},;:.]+", " ", text)
    if not keep_numbers:
        text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-Z0-9\- ]+", " ", text)
    text = text.replace("-", " ")
    tokens = []
    for token in text.split():
        token = TOKEN_REPLACEMENTS.get(token, token)
        if token in ROMAN_WORDS:
            continue
        tokens.append(token)
    return normalize_spaces(" ".join(tokens))


def normalize_role(text: str) -> str:
    return normalize_text(text, keep_numbers=True)


def role_tokens_without_fillers(text: str) -> List[str]:
    return [t for t in normalize_role(text).split() if t not in FILLER_WORDS]


def role_key_without_fillers(text: str) -> str:
    return " ".join(role_tokens_without_fillers(text))


EPI_WORD_REPLACEMENTS = {
    # Abreviações observadas no relatório do sistema.
    "PROT": "PROTECAO",
    "AG": "AGENTE",
    "MEC": "MECANICO",
    "TIP": "TIPO",
    "VAQUE": "VAQUETA",
    "ABRA": "ABRASIVO",
    "ESCO": "ESCORIANTES",
    # Singular/plural que não altera a identidade do EPI.
    "LUVAS": "LUVA",
    "AGENTES": "AGENTE",
    "MECANICOS": "MECANICO",
    "TERMICOS": "TERMICO",
    "ABRASIVOS": "ABRASIVO",
}

# Palavras puramente gramaticais que aparecem ou desaparecem nos cadastros
# sem mudar a identidade do EPI. Elas são removidas somente na normalização
# de EPI, não na normalização de cargos.
EPI_FILLER_WORDS = {
    "DE", "DA", "DO", "DAS", "DOS", "E", "CONTRA", "COM", "PARA", "TIPO"
}


def normalize_epi(text: str) -> str:
    """Normaliza nomes de EPI e aproxima abreviações reais do sistema.

    Exemplos que passam a ser equivalentes:
    - "LUVA PROTEÇÃO ANTICORTE" <-> "luvas proteção (anti corte)"
    - "LUVA DE PROTEÇÃO AGENTES MECANICOS VAQUETA/ RASPA (MISTA)"
      <-> "Luvas de prot. ag Mec. Vaque/Raspa(mista)"
    - "CALÇADO TIPO BOTINA (contra agentes abrasivos e escoriantes)"
      <-> "calçado tip botina c/ agente abra esco"

    A normalização é focada em abreviações e diferenças de flexão.
    Ela NÃO transforma categorias distintas (por exemplo, FACIAL e SOLAR)
    em equivalentes.
    """
    text = "" if text is None else str(text)
    # "c/" é uma abreviação muito comum no cadastro: c/ agente -> com agente.
    text = re.sub(r"\bC\s*/\s*", "COM ", strip_accents(text).upper())
    normalized = normalize_text(text, keep_numbers=True)

    # Variações de singular/plural e abreviações observadas.
    normalized = re.sub(r"\bANTI\s+CORTE\b", "ANTICORTE", normalized)
    tokens = []
    for token in normalized.split():
        token = EPI_WORD_REPLACEMENTS.get(token, token)
        if token in EPI_FILLER_WORDS:
            continue
        tokens.append(token)

    return normalize_spaces(" ".join(tokens))


# -----------------------------------------------------------------------------
# Cargos do PGR: algumas linhas agregam várias denominações.
# Ex.: "Pintor, Pintor I, Pintor II" -> "Pintor".
# A função gera candidatos para cruzar com o relatório.
# -----------------------------------------------------------------------------


def expand_role_aliases(role: str) -> List[str]:
    role = "" if role is None else str(role).strip()
    if not role:
        return []

    aliases = [role]

    # O que vem antes de parênteses normalmente é o cargo-base.
    before_paren = re.split(r"\(", role, maxsplit=1)[0].strip(" ,")
    if before_paren:
        aliases.append(before_paren)

    # Separação de listas agregadas.
    pieces = re.split(r"[,/;]+", before_paren)
    for piece in pieces:
        piece = piece.strip(" ,")
        if not piece:
            continue
        # Um item isolado "I, II e III" não vira cargo sozinho.
        if role_key_without_fillers(piece) in {"I", "II", "III", "IV"}:
            continue
        aliases.append(piece)

    # Também usa o primeiro segmento antes da vírgula como forte candidato.
    if pieces and pieces[0].strip():
        aliases.append(pieces[0].strip())

    clean = []
    seen = set()
    for alias in aliases:
        n = normalize_role(alias)
        if not n:
            continue
        for candidate in (n, role_key_without_fillers(alias)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                clean.append(candidate)
    return clean


# -----------------------------------------------------------------------------
# Leitura dos arquivos
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_report(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]

    expected = {"Projeto", "Cargo", "Risco", "EPI"}
    if not expected.issubset(df.columns):
        # Tenta localizar por posição, caso o arquivo venha com nomes ligeiramente diferentes.
        if len(df.columns) >= 4:
            df = df.iloc[:, :4].copy()
            df.columns = ["Projeto", "Cargo", "Risco", "EPI"]
        else:
            raise ValueError(
                "O relatório precisa ter as colunas Projeto, Cargo, Risco e EPI."
            )

    for col in ["Projeto", "Cargo", "Risco", "EPI"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def read_pgr(file_bytes: bytes) -> Tuple[pd.DataFrame, List[str]]:
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    # O modelo informado possui cabeçalho de EPI na linha 2 (índice 1)
    # e cargo na coluna B (índice 1).
    header_row = None
    for i in range(min(10, len(raw))):
        vals = raw.iloc[i].fillna("").astype(str).tolist()
        row_text = " ".join(vals).upper()
        nonempty = sum(bool(v.strip()) for v in vals)
        # No modelo do Anexo IV, a linha 1 contém os nomes dos EPIs e começa com ITEM.
        if "ITEM" in row_text and nonempty >= 5:
            header_row = i
            break
    if header_row is None:
        header_row = 1

    headers = raw.iloc[header_row].tolist()

    cargo_col = None
    for idx, h in enumerate(headers):
        hs = strip_accents(str(h)).upper() if h is not None else ""
        if idx == 1 or "CARGO" in hs or "FUNCAO" in hs:
            cargo_col = idx
            if idx == 1:
                break
    if cargo_col is None:
        cargo_col = 1

    epi_columns: Dict[int, str] = {}
    for idx, h in enumerate(headers):
        if idx <= cargo_col or h is None:
            continue
        name = normalize_spaces(str(h).replace("\n", " "))
        if not name or name.lower() == "nan":
            continue
        # Coluna final de controle / rodapé não é EPI.
        if idx >= raw.shape[1]:
            continue
        epi_columns[idx] = name

    records = []
    for r in range(header_row + 1, len(raw)):
        cargo = raw.iat[r, cargo_col] if cargo_col < raw.shape[1] else None
        if pd.isna(cargo) or not str(cargo).strip():
            continue
        cargo = str(cargo).strip()
        # Ignora linhas do rodapé/modelo sem cargo real.
        if cargo.upper() in {"CARGO/FUNÇÃO", "NAN"}:
            continue
        rec = {"ITEM": raw.iat[r, 0] if raw.shape[1] > 0 else "", "CARGO_PGR": cargo}
        for idx, epi in epi_columns.items():
            value = raw.iat[r, idx] if idx < raw.shape[1] else None
            rec[epi] = "" if pd.isna(value) else str(value).strip()
        records.append(rec)

    pgr = pd.DataFrame(records)
    if pgr.empty:
        raise ValueError("Não foi possível encontrar os cargos no Anexo IV.")

    # Só considera colunas que realmente aparecem como EPI no cabeçalho.
    epi_names = list(epi_columns.values())
    return pgr, epi_names


# -----------------------------------------------------------------------------
# Anexo I — Inventário de Riscos
# -----------------------------------------------------------------------------

def split_top_level(text: str) -> List[str]:
    """Divide a lista de cargos, respeitando vírgulas dentro de parênteses."""
    text = "" if text is None else str(text)
    parts = []
    current = []
    depth = 0

    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1

        if depth == 0 and ch in ";,":
            piece = "".join(current).strip(" .")
            if piece:
                parts.append(piece)
            current = []
        else:
            current.append(ch)

    piece = "".join(current).strip(" .")
    if piece:
        parts.append(piece)

    return parts


def find_text_in_rows(raw: pd.DataFrame, prefix: str, max_rows: int = 10) -> str:
    prefix_norm = strip_accents(prefix).upper()
    for r in range(min(max_rows, len(raw))):
        for c in range(raw.shape[1]):
            value = str(raw.iat[r, c]) if pd.notna(raw.iat[r, c]) else ""
            value_norm = strip_accents(value).upper().strip()
            if value_norm.startswith(prefix_norm):
                return value
    return ""


@st.cache_data(show_spinner=False)
def read_inventory(file_bytes: bytes) -> pd.DataFrame:
    """
    Lê todas as abas do Anexo I e transforma cada linha de risco em um registro.
    As funções do GHE/planilha são repetidas em cada registro para permitir
    o cruzamento Cargo × Risco com o relatório do sistema.
    """
    excel = pd.ExcelFile(io.BytesIO(file_bytes))
    records = []

    for sheet_name in excel.sheet_names:
        raw = pd.read_excel(excel, sheet_name=sheet_name, header=None)

        functions_cell = find_text_in_rows(raw, "Função(s):", max_rows=8)
        if not functions_cell:
            # Ex.: Planilha1 oculta com tabelas auxiliares.
            continue

        functions_text = re.sub(
            r"^\s*Função\(s\):\s*",
            "",
            functions_cell,
            flags=re.IGNORECASE,
        ).strip()

        sector_cell = find_text_in_rows(raw, "Setor:", max_rows=8)
        sector = re.sub(r"^\s*Setor:\s*", "", sector_cell, flags=re.IGNORECASE).strip()

        ghe_cell = find_text_in_rows(raw, "GHE:", max_rows=5)
        ghe = re.sub(r"^\s*GHE:\s*", "", ghe_cell, flags=re.IGNORECASE).strip()

        risk_header_row = None
        for r in range(min(15, len(raw))):
            row_text = " ".join(
                str(raw.iat[r, c]) if pd.notna(raw.iat[r, c]) else ""
                for c in range(raw.shape[1])
            )
            row_norm = strip_accents(row_text).upper()
            if (
                "PERIGO OU FATOR DE RISCO OCUPACIONAL" in row_norm
                and "RISCOS" in row_norm
            ):
                risk_header_row = r
                break

        if risk_header_row is None:
            continue

        for r in range(risk_header_row + 1, len(raw)):
            item = str(raw.iat[r, 0]).strip() if raw.shape[1] > 0 and pd.notna(raw.iat[r, 0]) else ""
            perigo = str(raw.iat[r, 2]).strip() if raw.shape[1] > 2 and pd.notna(raw.iat[r, 2]) else ""
            risco = str(raw.iat[r, 3]).strip() if raw.shape[1] > 3 and pd.notna(raw.iat[r, 3]) else ""
            tarefa = str(raw.iat[r, 1]).strip() if raw.shape[1] > 1 and pd.notna(raw.iat[r, 1]) else ""

            if not perigo and not risco:
                continue

            # Ignora rodapés e textos fora da tabela de riscos.
            if not item and not perigo:
                continue

            records.append({
                "Planilha Inventário": sheet_name,
                "GHE": ghe,
                "Setor": sector,
                "Funções no Inventário": functions_text,
                "Cargos individuais": " | ".join(split_top_level(functions_text)),
                "Item": item,
                "Tarefa/Fonte": tarefa,
                "Perigo/Fator de risco": perigo,
                "Risco detalhado": risco,
            })

    inventory = pd.DataFrame(records)
    if inventory.empty:
        raise ValueError(
            "Não foi possível localizar a tabela de riscos no Anexo I. "
            "Verifique se o arquivo possui as colunas 'Perigo ou fator de risco ocupacional' e 'Riscos'."
        )
    return inventory


def build_inventory_role_index(inventory: pd.DataFrame):
    """Relaciona cada cargo do Anexo I às planilhas/GHEs onde ele aparece."""
    alias_to_groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    grouped = inventory.groupby(
        ["Planilha Inventário", "GHE", "Setor", "Funções no Inventário"],
        dropna=False,
    ).size().reset_index(name="_n")

    for _, row in grouped.iterrows():
        functions = str(row["Funções no Inventário"])
        group_key = (
            str(row["Planilha Inventário"]),
            str(row["GHE"]),
            str(row["Setor"]),
            str(row["Funções no Inventário"]),
        )
        for role_entry in split_top_level(functions):
            for alias in expand_role_aliases(role_entry):
                if alias:
                    alias_to_groups[alias].append(
                        (
                            group_key,
                            100.0,
                        )
                    )

    return alias_to_groups, sorted(alias_to_groups.keys())


def match_inventory_groups(system_role: str, alias_to_groups, inventory_aliases):
    """
    Encontra o(s) GHE(s) do Anexo I correspondentes ao cargo do sistema.
    Usa primeiro igualdade normalizada; só depois aplica fuzzy com limiar
    conservador para não espalhar um cargo entre vários GHEs.
    """
    aliases = expand_role_aliases(system_role)
    matched = {}

    for alias in aliases:
        for group, score in alias_to_groups.get(alias, []):
            matched[group] = max(matched.get(group, 0.0), score)

    if not matched:
        for alias in aliases:
            if not alias:
                continue
            results = process.extract(
                alias,
                inventory_aliases,
                scorer=fuzz.token_set_ratio,
                limit=5,
            )
            for candidate, score, _ in results:
                score2 = fuzz.ratio(alias, candidate)
                final_score = max(float(score), float(score2))
                if final_score >= 86:
                    for group, _ in alias_to_groups.get(candidate, []):
                        # Um fuzzy muito fraco não deve acumular múltiplos GHEs.
                        matched[group] = max(matched.get(group, 0.0), final_score)

    return sorted(matched), sorted(
        matched.items(),
        key=lambda item: -item[1],
    )


RISK_MARKERS = [
    ("TRABALHO ESPAÇO CONFINADO", [
        "TRABALHO ESPACO CONFINADO",
        "ESPACO CONFINADO",
    ]),
    ("TRABALHO EM ALTURA", [
        "TRABALHO EM ALTURA",
    ]),
    ("CHOQUE ELÉTRICO", [
        "CHOQUE ELETRICO",
    ]),
    ("RUÍDO", [
        "RUIDO",
    ]),
    ("VIBRAÇÃO MÃO-BRAÇO", [
        "VIBRACAO DE MAOS E BRACO",
        "VIBRACAO MAOS E BRACO",
        "VIBRACOES LOCALIZADAS",
        "VIBRACAO MAO BRACO",
        "MAO BRACO",
        "VMB",
    ]),
    ("VIBRAÇÃO CORPO INTEIRO", [
        "VIBRACAO DE CORPO INTEIRO",
        "VIBRACAO CORPO INTEIRO",
        "VCI",
    ]),
    ("CALOR", [
        "CALOR",
    ]),
    ("RADIAÇÃO NÃO IONIZANTE", [
        "RADIACAO NAO IONIZANTE",
    ]),
    ("ARRANJO FÍSICO INADEQUADO", [
        "ARRANJO FISICO INADEQUADO",
    ]),
    ("MOVIMENTOS INERENTES À FUNÇÃO", [
        "MOVIMENTOS INERENTES A FUNCAO",
        "MOVIMENTOS INERENTES",
    ]),
    ("MOVIMENTOS REPETITIVOS", [
        "MOVIMENTOS REPETITIVOS",
    ]),
    ("ESTRESSE ORGANIZACIONAL", [
        "ESTRESSE ORGANIZACIONAL",
    ]),
    ("ACETONA", [
        "ACETONA",
    ]),
    ("GRAXA / ÓLEO MINERAL", [
        "GRAXA A BASE DE OLEO MINERAL",
        "GRAXA A BASE DE OLEO",
        "OLEO MINERAL",
    ]),
    ("FUMOS METÁLICOS — CÁDMIO", [
        "FUMOS METALICOS CADMIO",
        "FUMOS MATALICOS CADMIO",
    ]),
    ("FUMOS METÁLICOS — MANGANÊS", [
        "FUMOS METALICOS MANGANES",
        "FUMOS MATALICOS MANGANES",
    ]),
    ("FUMOS METÁLICOS — ÓXIDO DE FERRO", [
        "FUMOS METALICOS OXIDO DE FERRO",
        "FUMOS MATALICOS OXIDO DE FERRO",
    ]),
    ("FUMOS METÁLICOS — COBRE", [
        "FUMOS METALICOS COBRE",
        "FUMOS MATALICOS COBRE",
    ]),
    ("FUMOS METÁLICOS — CHUMBO", [
        "FUMOS METALICOS CHUMBO",
        "FUMOS MATALICOS CHUMBO",
    ]),
    ("TOLUENO", ["TOLUENO"]),
    ("XILENO", ["XILENO"]),
    ("HEXANO", ["HEXANO"]),
    ("BENZENO", ["BENZENO"]),
    ("HIDROCARBONETOS", [
        "HIDROCARBONETO",
        "HIDROCARBONETOS",
    ]),
    ("POEIRA RESPIRÁVEL", [
        "POEIRA RESPIRAVEL",
    ]),
    ("POEIRA MINERAL", [
        "POEIRA MINERAL",
    ]),
    ("AGENTES BIOLÓGICOS", [
        "AGENTES INFECCIOSOS",
        "VIRUS BACTERIAS PATOGENICAS",
        "VIRUS BACTERIAS",
    ]),
]


def normalize_risk_text(text: str) -> str:
    text = strip_accents("" if text is None else str(text)).upper()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\n", " ")
    text = re.sub(r"[()\[\]{},;:/]+", " ", text)
    text = re.sub(r"[^A-Z0-9\- ]+", " ", text)
    text = text.replace("-", " ")
    return normalize_spaces(text)


def risk_family(text: str) -> str:
    normalized = normalize_risk_text(text)

    if "ERGONOM" in normalized or "PSICOSSOC" in normalized or "BIOMECAN" in normalized:
        return "ERGONÔMICO"
    if "FISICO" in normalized:
        return "FÍSICO"
    if "QUIMICO" in normalized:
        return "QUÍMICO"
    if "BIOLOGICO" in normalized:
        return "BIOLÓGICO"
    if any(word in normalized for word in ["ACIDENT", "MECANICO", "MECANICOS"]):
        return "ACIDENTE"
    return "OUTRO"


def risk_marker(text: str) -> str:
    normalized = normalize_risk_text(text)

    for canonical, aliases in sorted(
        RISK_MARKERS,
        key=lambda item: max(len(alias) for alias in item[1]),
        reverse=True,
    ):
        if any(alias in normalized for alias in aliases):
            return canonical

    if "NAO DETECTADO" in normalized or "NAO IDENTIFICADO" in normalized:
        return "NÃO IDENTIFICADO"

    return ""


def compare_risk(system_risk: str, inventory_row: pd.Series):
    """
    Compara um risco do relatório com uma linha do Anexo I.
    A comparação é propositalmente mais rígida que um fuzzy genérico:
    família de risco e marcador precisam ser compatíveis.
    """
    system_family = risk_family(system_risk)
    inventory_text = (
        f"{inventory_row.get('Perigo/Fator de risco', '')} "
        f"{inventory_row.get('Risco detalhado', '')}"
    )
    inventory_family = risk_family(inventory_text)

    if system_family == "OUTRO" or inventory_family == "OUTRO":
        return False, 0.0, ""

    if system_family != inventory_family:
        return False, 0.0, ""

    system_marker = risk_marker(system_risk)
    inventory_marker = risk_marker(inventory_text)

    if system_marker and inventory_marker:
        if system_marker == inventory_marker:
            return True, 100.0, "MESMO MARCADOR"

        # O relatório pode usar "Hidrocarboneto" de forma genérica enquanto
        # o inventário detalha Tolueno/Xileno/Hexano/Benzeno, e vice-versa.
        hydrocarbon_markers = {
            "HIDROCARBONETOS",
            "TOLUENO",
            "XILENO",
            "HEXANO",
            "BENZENO",
        }
        if system_marker == "HIDROCARBONETOS" and inventory_marker in hydrocarbon_markers:
            return True, 90.0, "FAMÍLIA HIDROCARBONETOS"
        if inventory_marker == "HIDROCARBONETOS" and system_marker in hydrocarbon_markers:
            return True, 88.0, "FAMÍLIA HIDROCARBONETOS"

        return False, 0.0, ""

    # Para casos em que não conseguimos extrair um marcador específico,
    # exigimos sobreposição textual forte dentro da mesma família.
    stopwords = {
        "FISICO", "QUIMICO", "BIOLOGICO", "ERGONOMICO",
        "PSICOSSOCIAIS", "COGNITIVOS", "BIOMECANICOS",
        "MECANICOS", "ACIDENTES", "ACIDENTE",
        "NAO", "DETECTADO", "IDENTIFICADO",
    }

    system_tokens = {
        token for token in normalize_risk_text(system_risk).split()
        if token not in stopwords
    }
    inventory_tokens = {
        token for token in normalize_risk_text(inventory_text).split()
        if token not in stopwords
    }

    overlap = system_tokens & inventory_tokens
    if len(overlap) >= 2:
        return True, 85.0, "TOKENS COMPARTILHADOS"

    return False, 0.0, ""


def analyze_risks(report: pd.DataFrame, inventory: pd.DataFrame):
    """
    Faz a auditoria bidirecional:
    1) todo risco do Anexo I precisa aparecer no sistema;
    2) riscos existentes no sistema que não aparecem no Anexo I são apontados
       como extras para conferência.
    """
    alias_to_groups, inventory_aliases = build_inventory_role_index(inventory)

    system_roles = sorted(
        str(role).strip()
        for role in report["Cargo"].dropna().unique()
        if str(role).strip()
    )

    system_risks_by_role: Dict[str, List[str]] = defaultdict(list)
    for _, row in report.iterrows():
        role = str(row["Cargo"]).strip()
        risk = str(row["Risco"]).strip()
        if role and risk:
            system_risks_by_role[role].append(risk)

    for role in list(system_risks_by_role):
        system_risks_by_role[role] = sorted(set(system_risks_by_role[role]))

    detail_rows = []
    missing_rows = []
    extra_rows = []
    missing_role_rows = []

    all_groups = inventory.groupby(
        ["Planilha Inventário", "GHE", "Setor", "Funções no Inventário"],
        dropna=False,
    ).size().reset_index(name="_n")

    # Mapa de grupo -> índices de linhas de risco.
    group_risk_rows = defaultdict(list)
    for idx, row in inventory.iterrows():
        key = (
            str(row["Planilha Inventário"]),
            str(row["GHE"]),
            str(row["Setor"]),
            str(row["Funções no Inventário"]),
        )
        group_risk_rows[key].append(idx)

    matched_roles_info = []

    for role in system_roles:
        matched_groups, group_scores = match_inventory_groups(
            role,
            alias_to_groups,
            inventory_aliases,
        )

        if not matched_groups:
            missing_role_rows.append({
                "Cargo no Sistema": role,
                "Status": "CARGO NÃO ENCONTRADO NO ANEXO I",
            })
            matched_roles_info.append({
                "Cargo no Sistema": role,
                "GHE/Planilha": "",
                "Setor": "",
                "Confiança": 0.0,
                "Status": "CARGO NÃO ENCONTRADO",
            })

            # Sem cargo correspondente no inventário, não classificamos os
            # riscos como "extras": simplesmente não existe uma base de comparação.
            continue

        role_conf = max(score for _, score in group_scores) if group_scores else 100.0

        matched_inventory_indexes = [
            idx
            for group_key in matched_groups
            for idx in group_risk_rows.get(tuple(group_key), [])
        ]
        matched_inventory = [inventory.loc[idx] for idx in matched_inventory_indexes]

        group_labels = []
        for group_key in matched_groups:
            group_label = " — ".join(
                part for part in tuple(group_key)[:3] if part and part != "nan"
            )
            group_labels.append(group_label)

        matched_roles_info.append({
            "Cargo no Sistema": role,
            "GHE/Planilha": " | ".join(group_labels),
            "Setor": " | ".join(sorted({
                str(row["Setor"]).strip()
                for row in matched_inventory
                if str(row["Setor"]).strip() and str(row["Setor"]).strip() != "nan"
            })),
            "Confiança": round(role_conf, 1),
            "Status": "ATENDIDO",
        })

        # Para cada risco do sistema, encontra o melhor risco do Anexo I.
        used_inventory_indexes = set()
        for system_risk in system_risks_by_role.get(role, []):
            best_match = None

            for idx in matched_inventory_indexes:
                inv_row = inventory.loc[idx]
                ok, score, reason = compare_risk(system_risk, inv_row)
                if not ok:
                    continue

                candidate = (inv_row, score, reason, idx)
                if best_match is None or score > best_match[1]:
                    best_match = candidate

            if best_match is None:
                extra_rows.append({
                    "Cargo no Sistema": role,
                    "GHE/Planilha": " | ".join(group_labels),
                    "Setor": " | ".join(sorted({
                        str(row["Setor"]).strip()
                        for row in matched_inventory
                        if str(row["Setor"]).strip() and str(row["Setor"]).strip() != "nan"
                    })),
                    "Risco do Sistema": system_risk,
                    "Família": risk_family(system_risk),
                    "Status": "EXTRA NO SISTEMA",
                })
                continue

            inv_row, score, reason, idx = best_match
            used_inventory_indexes.add(idx)
            status = "ATENDIDO" if score >= 92 else "ATENDIDO — CORRESPONDÊNCIA PROVÁVEL"

            detail_rows.append({
                "Cargo no Sistema": role,
                "GHE/Planilha": " | ".join(group_labels),
                "Setor": str(inv_row["Setor"]).strip(),
                "Confiança cargo": round(role_conf, 1),
                "Risco do Sistema": system_risk,
                "Família": risk_family(system_risk),
                "Risco/Perigo no Anexo I": str(inv_row["Perigo/Fator de risco"]).strip(),
                "Risco detalhado no Anexo I": str(inv_row["Risco detalhado"]).strip(),
                "Confiança risco": score,
                "Status": status,
            })

        # Agora o caminho inverso: todo risco do inventário deve aparecer no sistema.
        for idx in matched_inventory_indexes:
            inv_row = inventory.loc[idx]
            inventory_text = (
                f"{inv_row['Perigo/Fator de risco']} "
                f"{inv_row['Risco detalhado']}"
            )
            matched_any = False
            best_system = None

            for system_risk in system_risks_by_role.get(role, []):
                ok, score, reason = compare_risk(system_risk, inv_row)
                if ok:
                    matched_any = True
                    candidate = (system_risk, score, reason)
                    if best_system is None or score > best_system[1]:
                        best_system = candidate

            if not matched_any:
                missing_rows.append({
                    "Cargo no Sistema": role,
                    "GHE/Planilha": " | ".join(group_labels),
                    "Setor": str(inv_row["Setor"]).strip(),
                    "Item": str(inv_row["Item"]).strip(),
                    "Perigo/Fator de risco": str(inv_row["Perigo/Fator de risco"]).strip(),
                    "Risco detalhado": str(inv_row["Risco detalhado"]).strip(),
                    "Status": "FALTA NO SISTEMA",
                })

    detail_df = pd.DataFrame(detail_rows)
    missing_df = pd.DataFrame(missing_rows)
    extras_df = pd.DataFrame(extra_rows)
    missing_roles_df = pd.DataFrame(missing_role_rows)
    matched_roles_df = pd.DataFrame(matched_roles_info)

    inventory_roles = set()
    for functions_text in inventory["Funções no Inventário"].dropna().astype(str):
        for role_entry in split_top_level(functions_text):
            base = re.split(r"\(", role_entry, maxsplit=1)[0].strip(" ,.")
            if base:
                inventory_roles.add(normalize_role(base))

    risk_summary = {
        "cargos_inventario": int(len(inventory_roles)),
        "cargos_sistema_risco": int(len(system_roles)),
        "cargos_sem_inventario": int(len(missing_roles_df)),
        "riscos_atendidos": int(
            detail_df["Status"].isin(["ATENDIDO", "ATENDIDO — CORRESPONDÊNCIA PROVÁVEL"]).sum()
        ) if not detail_df.empty else 0,
        "riscos_em_falta": int(len(missing_df)),
        "riscos_extras_sistema": int(len(extras_df)),
    }

    return (
        risk_summary,
        matched_roles_df,
        detail_df,
        missing_df,
        extras_df,
        missing_roles_df,
    )


# -----------------------------------------------------------------------------
# Matching de cargos
# -----------------------------------------------------------------------------


def build_report_role_index(report: pd.DataFrame):
    normalized_to_original: Dict[str, List[str]] = defaultdict(list)
    no_fillers_to_original: Dict[str, List[str]] = defaultdict(list)

    for role in sorted(report["Cargo"].dropna().unique()):
        role = str(role).strip()
        if not role:
            continue
        n = normalize_role(role)
        nf = role_key_without_fillers(role)
        if n:
            normalized_to_original[n].append(role)
        if nf:
            no_fillers_to_original[nf].append(role)
    return normalized_to_original, no_fillers_to_original


def match_report_roles(pgr_role: str, report_roles: List[str], normalized_index, no_fillers_index):
    aliases = expand_role_aliases(pgr_role)
    matched = set()
    best_candidates = []

    for alias in aliases:
        if alias in normalized_index:
            matched.update(normalized_index[alias])
        if alias in no_fillers_index:
            matched.update(no_fillers_index[alias])

    # Se não achou por igualdade, procura fuzzy por cada alias.
    if not matched:
        for alias in aliases:
            if not alias:
                continue
            choices = list({normalize_role(r): r for r in report_roles}.keys())
            results = process.extract(alias, choices, scorer=fuzz.token_set_ratio, limit=5)
            for norm_candidate, score, _ in results:
                original = next(r for r in report_roles if normalize_role(r) == norm_candidate)
                # Regras extras para reduzir falsos positivos.
                score2 = fuzz.ratio(alias, norm_candidate)
                final_score = max(score, score2)
                if final_score >= 86:
                    matched.add(original)
                    best_candidates.append((original, final_score))

    # Remove variantes duplicadas e guarda a maior pontuação.
    score_map = {}
    for role, score in best_candidates:
        score_map[role] = max(score_map.get(role, 0), score)
    for role in matched:
        score_map.setdefault(role, 100.0 if normalize_role(role) in aliases else 90.0)

    return sorted(matched), sorted(score_map.items(), key=lambda x: -x[1])


# -----------------------------------------------------------------------------
# Matching de EPIs
# -----------------------------------------------------------------------------


def epi_compatible(required: str, candidate: str) -> bool:
    r = normalize_epi(required)
    c = normalize_epi(candidate)

    # Regras de identidade muito específicas. Elas vêm antes do fuzzy para
    # evitar falsas equivalências entre EPIs de famílias diferentes.
    # Ex.: "PROTETOR FACIAL" não pode ser atendido por "PROTETOR SOLAR".
    if "PROTETOR FACIAL" in r:
        return "FACIAL" in c and "SOLAR" not in c
    if "PROTETOR SOLAR" in r:
        return "SOLAR" in c and "FACIAL" not in c
    if "PROTETOR AUDITIVO" in r:
        return "AUDITIVO" in c

    # Características que diferenciam materiais/modelos.
    required_markers = [
        ("PU", ["PU"]),
        ("PVC", ["PVC"]),
        ("NITRILICA", ["NITRILICA"]),
        ("ANTICORTE", ["ANTICORTE"]),
        ("ISOLANTE", ["ISOLANTE"]),
        ("TYCHEM", ["TYCHEM"]),
        ("PFF3", ["PFF3"]),
        ("PFF2", ["PFF2"]),
        ("UVEX", ["UVEX"]),
        ("INCOLOR", ["INCOLOR"]),
        ("MAÇAR", ["MACAR", "MACARIQUEIRO"]),
        ("TONALIDADE 5", ["TONALIDADE 5"]),
        ("PLUG", ["PLUG"]),
        ("CONCHA", ["CONCHA"]),
        ("CHUVA", ["CHUVA"]),
        ("BARBEIRO", ["BARBEIRO"]),
        ("BOTA", ["BOTA"]),
        ("BOTINA", ["BOTINA"]),
        ("FACIAL INTEIRA", ["FACIAL INTEIRA"]),
        ("PECA SEMIFAC", ["SEMIFAC", "SEMI FAC"]),
        ("FILTRO PARA MASCARA", ["FILTRO"]),
    ]

    for trigger, accepted in required_markers:
        if trigger in r and not any(a in c for a in accepted):
            return False

    # TALABARTE: o PGR pode exigir especificamente o modelo DUPLO.
    # Nesse caso, um "Talabarte de Segurança" genérico NÃO atende.
    # Um "Talabarte de Segurança Duplo Retrátil", por exemplo, atende,
    # pois mantém a característica obrigatória "DUPLO" e apenas acrescenta
    # uma especificação adicional (RETRÁTIL).
    if "TALABARTE" in r:
        if "DUPLO" in r and "DUPLO" not in c:
            return False
        if "DUPLO" not in r and "DUPLO" in c:
            # Um talabarte duplo pode ser mais específico que o genérico.
            # Mantemos a compatibilidade nesse sentido.
            pass

    # Luvas: além da categoria LUVA, preserva características importantes do
    # modelo exigido. Isso evita que uma luva genérica atenda outra finalidade.
    if "LUVA" in r:
        if "ANTICORTE" in r and "ANTICORTE" not in c:
            return False
        if "NITRILICA" in r and "NITRILICA" not in c:
            return False
        if "PVC" in r and "PVC" not in c:
            return False
        if "PU" in r and "PU" not in c:
            return False
        if "TERMICO" in r and "TERMICO" not in c:
            return False
        if "VAQUETA" in r and "VAQUETA" not in c:
            return False
        if "RASPA" in r and "RASPA" not in c:
            return False

    # Botina e bota são categorias diferentes. Uma não atende automaticamente
    # a outra só porque ambas são calçados.
    if "BOTINA" in r and "BOTINA" not in c:
        return False
    if "BOTA" in r and "BOTINA" not in r and "BOTA" not in c:
        return False

    # Para EPIs com nome-base muito característico, exige que o mesmo item apareça.
    # Isso evita que um fuzzy genérico transforme "MANGOTE" em "CAPACETE", por exemplo.
    anchor_groups = [
        ("CALCADO", ["BOTA", "BOTINA"]),
        ("CAPACETE", ["CAPACETE"]),
        ("CINTO", ["CINTO"]),
        ("LUVA", ["LUVA", "LUVAS"]),
        ("MACACAO", ["MACACAO"]),
        ("MANGOTE", ["MANGOTE"]),
        ("MASCARA DE SOLDA", ["MASCARA", "SOLDA"]),
        ("OCULOS", ["OCULOS"]),
        ("PERNEIRA", ["PERNEIRA"]),
        ("PROTETOR", ["PROTETOR"]),
        ("RESPIRADOR", ["RESPIRADOR"]),
        ("FILTRO", ["FILTRO"]),
        ("TALABARTE", ["TALABARTE"]),
        ("VESTIMENTA", ["VESTIMENTA", "AVENTAL", "CAPA"]),
        ("ARMACAO", ["ARMACAO", "OCULOS"]),
    ]
    token_r = set(r.split())
    token_c = set(c.split())
    for trigger, accepted in anchor_groups:
        if trigger in r or trigger in token_r:
            if trigger == "CALCADO":
                # BOTA e BOTINA são categorias diferentes.
                if "BOTA" in r and "BOTINA" not in r:
                    if "BOTA" not in token_c:
                        return False
                elif "BOTINA" in r and "BOTINA" not in token_c:
                    return False
            elif not any(a in token_c for a in accepted):
                return False

    # Variantes com RASPA precisam respeitar a finalidade.
    if "TERMICO" in r and "TERMICO" not in c:
        return False
    if "AVENTAL" in r and "RASPA" in r and "RASPA" not in c:
        return False
    if "VAQUETA" in r or ("RASPA" in r and "MISTA" in r):
        if "VAQUETA" not in c and "RASPA" not in c:
            return False
    if "MACAR" in r and "MACAR" not in c and "TONALIDADE 5" not in c:
        return False

    return True


def match_epi(required_epi: str, report_epis: List[str]) -> Tuple[bool, str, float]:
    r = normalize_epi(required_epi)
    if not r:
        return False, "", 0.0

    candidates = [e for e in report_epis if e and str(e).strip().lower() != "sem nada"]

    # No PGR este EPI é um conjunto (calça + camisa); o relatório pode trazer as duas peças separadamente.
    if "CALCA E CAMISA" in r and "ELETRICISTA" in r:
        has_pants = any("CALCA" in normalize_epi(e) and "ELETRICISTA" in normalize_epi(e) for e in candidates)
        has_shirt = any("CAMISA" in normalize_epi(e) and "ELETRICISTA" in normalize_epi(e) for e in candidates)
        if has_pants and has_shirt:
            return True, "Calça eletricista + camisa eletricista", 100.0

    # 1) Igualdade normalizada
    for e in candidates:
        c = normalize_epi(e)
        if r == c:
            return True, e, 100.0

    # 2) Casos com siglas / descrições mais longas.
    best = None
    for e in candidates:
        if not epi_compatible(required_epi, e):
            continue
        c = normalize_epi(e)
        score_set = fuzz.token_set_ratio(r, c)
        score_partial = fuzz.partial_ratio(r, c)
        score = max(score_set, score_partial)

        # Regras específicas do vocabulário observado no relatório.
        # "Talabarte de Segurança Duplo" exige que o candidato também tenha
        # a característica "DUPLO". Assim evitamos aceitar um talabarte genérico.
        if "TALABARTE" in r and "DUPLO" in r:
            if "DUPLO" not in c:
                continue
            # Se o candidato tiver palavras adicionais, como RETRÁTIL, isso
            # é aceito; não reduzimos a pontuação por ser mais específico.
            if "RETRATIL" in c:
                score += 5
        if "TONALIDADE 5" in r and "TONALIDADE 5" in c:
            score += 12
        if "TONALIDADE 5" in r and "MAÇAR" in r and "MAÇAR" not in c:
            score -= 8
        if "FILTRO PARA MASCARA" in r and "FILTRO PARA MASCARA" in c:
            score += 8
        if "CALCADO TIPO BOTINA" in r and "BOTINA" in c:
            score += 8
        if "CALCADO TIPO BOTA" in r and "BOTA" in c and "BOTINA" not in c:
            score += 8

        # A confiança é exibida em escala de 0 a 100.
        score = min(float(score), 100.0)

        if best is None or score > best[1]:
            best = (e, score)

    if best is None:
        return False, "", 0.0

    # 92+: alta confiança; 78–91: possível correspondência.
    if best[1] >= 78:
        return True, best[0], round(float(best[1]), 1)
    return False, best[0], round(float(best[1]), 1)


# -----------------------------------------------------------------------------
# Análise
# -----------------------------------------------------------------------------


def analyze(report: pd.DataFrame, pgr: pd.DataFrame, epi_names: List[str]):
    report_roles = sorted(report["Cargo"].dropna().unique().tolist())
    report_epi_by_role: Dict[str, List[str]] = defaultdict(list)
    for _, row in report.iterrows():
        role = str(row["Cargo"]).strip()
        epi = str(row["EPI"]).strip()
        if role and epi and epi.lower() != "sem nada":
            report_epi_by_role[role].append(epi)
    for role in list(report_epi_by_role):
        report_epi_by_role[role] = sorted(set(report_epi_by_role[role]))

    normalized_index, no_fillers_index = build_report_role_index(report)

    detailed = []
    matched_role_info = []
    missing_role_rows = []

    for _, row in pgr.iterrows():
        pgr_role = str(row["CARGO_PGR"]).strip()
        required = []
        for epi in epi_names:
            val = row.get(epi, "")
            if pd.notna(val) and str(val).strip():
                required.append((epi, str(val).strip()))

        matched_roles, role_scores = match_report_roles(
            pgr_role, report_roles, normalized_index, no_fillers_index
        )

        if not matched_roles:
            missing_role_rows.append({
                "Cargo PGR": pgr_role,
                "Status": "CARGO NÃO ENCONTRADO NO RELATÓRIO",
            })
            for epi, mark in required:
                detailed.append({
                    "Cargo PGR": pgr_role,
                    "Cargo(s) no Sistema": "",
                    "Confiança cargo": 0,
                    "EPI do PGR": epi,
                    "Marca PGR": mark,
                    "EPI encontrado no Sistema": "",
                    "Confiança EPI": 0,
                    "Status": "FALTA — CARGO NÃO ENCONTRADO",
                })
            matched_role_info.append({
                "Cargo PGR": pgr_role,
                "Cargo(s) no Sistema": "",
                "Confiança": 0,
                "Status": "CARGO NÃO ENCONTRADO",
            })
            continue

        role_conf = max(score for _, score in role_scores) if role_scores else 100.0
        system_epis = sorted({epi for r in matched_roles for epi in report_epi_by_role.get(r, [])})
        role_has_missing = False

        for epi, mark in required:
            found, system_epi, epi_score = match_epi(epi, system_epis)
            if found:
                status = "OK" if epi_score >= 92 else "OK — CORRESPONDÊNCIA PROVÁVEL"
            else:
                status = "FALTA"
                role_has_missing = True

            detailed.append({
                "Cargo PGR": pgr_role,
                "Cargo(s) no Sistema": " | ".join(matched_roles),
                "Confiança cargo": round(role_conf, 1),
                "EPI do PGR": epi,
                "Marca PGR": mark,
                "EPI encontrado no Sistema": system_epi,
                "Confiança EPI": epi_score,
                "Status": status,
            })

        matched_role_info.append({
            "Cargo PGR": pgr_role,
            "Cargo(s) no Sistema": " | ".join(matched_roles),
            "Confiança": round(role_conf, 1),
            "Status": "FALHAS DE EPI" if role_has_missing else "ATENDIDO",
        })

    detail_df = pd.DataFrame(detailed)
    roles_df = pd.DataFrame(matched_role_info)
    missing_roles_df = pd.DataFrame(missing_role_rows)

    if detail_df.empty:
        missing_epi_df = pd.DataFrame()
    else:
        missing_epi_df = detail_df[detail_df["Status"].str.startswith("FALTA")].copy()

    # EPIs que existem no sistema mas não são exigidos pelo PGR daquele cargo.
    extras = []
    for _, r in roles_df.iterrows():
        cargo_pgr = r["Cargo PGR"]
        matched = [x.strip() for x in str(r["Cargo(s) no Sistema"]).split("|") if x.strip()]
        if not matched:
            continue
        system_epis = sorted({epi for role in matched for epi in report_epi_by_role.get(role, [])})
        required_epis = detail_df.loc[detail_df["Cargo PGR"] == cargo_pgr, "EPI do PGR"].tolist()
        required_norm = {normalize_epi(x) for x in required_epis}
        for epi in system_epis:
            # Só sinaliza como extra quando nenhum EPI do PGR bate com ele.
            found_as_required = any(
                match_epi(req, [epi])[0] for req in required_epis
            )
            if not found_as_required and normalize_epi(epi) not in required_norm:
                extras.append({
                    "Cargo PGR": cargo_pgr,
                    "Cargo(s) no Sistema": " | ".join(matched),
                    "EPI extra no Sistema": epi,
                })
    extras_df = pd.DataFrame(extras)

    summary = {
        "cargos_pgr": int(len(pgr)),
        "cargos_atendidos": int((roles_df["Status"] == "ATENDIDO").sum()) if not roles_df.empty else 0,
        "cargos_com_falha": int((roles_df["Status"] == "FALHAS DE EPI").sum()) if not roles_df.empty else 0,
        "cargos_nao_encontrados": int(len(missing_roles_df)),
        "itens_epi_pgr": int(len(detail_df)),
        "epis_em_falta": int(len(missing_epi_df)),
        "epis_extra": int(len(extras_df)),
    }

    return summary, roles_df, detail_df, missing_epi_df, missing_roles_df, extras_df


# -----------------------------------------------------------------------------
# Exportação
# -----------------------------------------------------------------------------


def build_excel(
    summary,
    roles_df,
    detail_df,
    missing_epi_df,
    missing_roles_df,
    extras_df,
    risk_roles_df,
    risk_detail_df,
    risk_missing_df,
    risk_extras_df,
    risk_missing_roles_df,
):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Resumo", index=False)
        roles_df.to_excel(writer, sheet_name="Cargos", index=False)
        detail_df.to_excel(writer, sheet_name="Detalhado EPI", index=False)
        missing_epi_df.to_excel(writer, sheet_name="Faltas EPI", index=False)
        missing_roles_df.to_excel(writer, sheet_name="Cargos ausentes EPI", index=False)
        extras_df.to_excel(writer, sheet_name="Extras EPI", index=False)
        risk_roles_df.to_excel(writer, sheet_name="Cargos x Inventário", index=False)
        risk_detail_df.to_excel(writer, sheet_name="Detalhado Riscos", index=False)
        risk_missing_df.to_excel(writer, sheet_name="Faltas Riscos", index=False)
        risk_extras_df.to_excel(writer, sheet_name="Extras Riscos", index=False)
        risk_missing_roles_df.to_excel(writer, sheet_name="Cargos ausentes Risco", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#17365D",
            "font_color": "white",
            "border": 1,
        })
        ok_fmt = workbook.add_format({"bg_color": "#E2F0D9"})
        fail_fmt = workbook.add_format({"bg_color": "#FCE4D6"})
        probable_fmt = workbook.add_format({"bg_color": "#FFF2CC"})

        sheet_frames = {
            "Resumo": pd.DataFrame([summary]),
            "Cargos": roles_df,
            "Detalhado EPI": detail_df,
            "Faltas EPI": missing_epi_df,
            "Cargos ausentes EPI": missing_roles_df,
            "Extras EPI": extras_df,
            "Cargos x Inventário": risk_roles_df,
            "Detalhado Riscos": risk_detail_df,
            "Faltas Riscos": risk_missing_df,
            "Extras Riscos": risk_extras_df,
            "Cargos ausentes Risco": risk_missing_roles_df,
        }

        for sheet_name, df in sheet_frames.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)

            if len(df.columns) > 0:
                last_row = max(0, len(df))
                last_col = len(df.columns) - 1
                ws.autofilter(0, 0, last_row, last_col)

            for idx, col in enumerate(df.columns):
                width = min(max(len(str(col)) + 2, 14), 45)
                if not df.empty:
                    sample = df[col].astype(str).head(100)
                    width = min(max(width, int(sample.map(len).max()) + 2), 60)
                ws.set_column(idx, idx, width)
                ws.write(0, idx, col, header_fmt)

        if not detail_df.empty and "Status" in detail_df.columns:
            ws = writer.sheets["Detalhado EPI"]
            status_col = detail_df.columns.get_loc("Status")
            for row_idx, val in enumerate(detail_df["Status"].astype(str), start=1):
                if val.startswith("OK"):
                    ws.write(row_idx, status_col, val, ok_fmt)
                elif val.startswith("FALTA"):
                    ws.write(row_idx, status_col, val, fail_fmt)
                elif "PROVÁVEL" in val:
                    ws.write(row_idx, status_col, val, probable_fmt)

        if not risk_detail_df.empty and "Status" in risk_detail_df.columns:
            ws = writer.sheets["Detalhado Riscos"]
            status_col = risk_detail_df.columns.get_loc("Status")
            for row_idx, val in enumerate(risk_detail_df["Status"].astype(str), start=1):
                if val == "ATENDIDO":
                    ws.write(row_idx, status_col, val, ok_fmt)
                elif "PROVÁVEL" in val:
                    ws.write(row_idx, status_col, val, probable_fmt)

        if not risk_missing_df.empty:
            ws = writer.sheets["Faltas Riscos"]
            status_col = risk_missing_df.columns.get_loc("Status")
            for row_idx, val in enumerate(risk_missing_df["Status"].astype(str), start=1):
                ws.write(row_idx, status_col, val, fail_fmt)

        if not risk_extras_df.empty:
            ws = writer.sheets["Extras Riscos"]
            status_col = risk_extras_df.columns.get_loc("Status")
            for row_idx, val in enumerate(risk_extras_df["Status"].astype(str), start=1):
                ws.write(row_idx, status_col, val, probable_fmt)

    output.seek(0)
    return output.getvalue()


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

st.title("🦺 Auditor de EPIs e Riscos — PGR × Sistema")
st.markdown(
    "O aplicativo cruza **3 arquivos**: o relatório do sistema, o Anexo IV (Inventário de EPIs) "
    "e o Anexo I (Inventário de Riscos). Ele verifica tanto o atendimento dos EPIs quanto dos riscos."
)

with st.sidebar:
    st.header("Arquivos")
    st.caption("Envie os 3 arquivos para realizar a auditoria completa.")

    report_file = st.file_uploader(
        "1. Relatório do sistema",
        type=["xlsx", "xls"],
        key="report",
    )

    pgr_file = st.file_uploader(
        "2. Anexo IV — Inventário de EPIs",
        type=["xlsx", "xls"],
        key="pgr",
    )

    inventory_file = st.file_uploader(
        "3. Anexo I — Inventário de Riscos",
        type=["xlsx", "xls"],
        key="inventory",
    )

    st.divider()
    st.caption(
        "EPIs: qualquer célula preenchida no cruzamento Cargo × EPI do Anexo IV é tratada como EPI exigido. "
        "Riscos: o aplicativo compara os riscos do relatório com os perigos/riscos registrados no Anexo I, "
        "considerando família e marcadores específicos para evitar falsos positivos."
    )

if not report_file or not pgr_file or not inventory_file:
    st.info("Envie os três arquivos na barra lateral para iniciar a auditoria completa.")
    st.stop()

try:
    report = read_report(report_file.getvalue())
    pgr, epi_names = read_pgr(pgr_file.getvalue())
    inventory = read_inventory(inventory_file.getvalue())
except Exception as exc:
    st.error(f"Não foi possível ler os arquivos: {exc}")
    st.stop()

with st.spinner("Cruzando cargos, EPIs e riscos..."):
    (
        summary_epi,
        roles_df,
        detail_df,
        missing_epi_df,
        missing_roles_df,
        extras_df,
    ) = analyze(report, pgr, epi_names)

    (
        summary_risk,
        risk_roles_df,
        risk_detail_df,
        risk_missing_df,
        risk_extras_df,
        risk_missing_roles_df,
    ) = analyze_risks(report, inventory)

summary = {**summary_epi, **summary_risk}

st.subheader("🦺 Auditoria de EPIs")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cargos no Anexo IV", summary["cargos_pgr"])
m2.metric("Cargos atendidos", summary["cargos_atendidos"])
m3.metric("Cargos com falta EPI", summary["cargos_com_falha"])
m4.metric("Cargos não encontrados", summary["cargos_nao_encontrados"])
m5.metric("EPIs em falta", summary["epis_em_falta"])

if summary["epis_em_falta"] == 0 and summary["cargos_nao_encontrados"] == 0:
    st.success("EPIs: nenhuma falta foi encontrada pelo critério automático da análise.")
else:
    st.warning("EPIs: existem cargos ou EPIs do Anexo IV que não foram localizados automaticamente no sistema.")

if not detail_df.empty:
    status_options = st.multiselect(
        "Status EPI",
        ["OK", "OK — CORRESPONDÊNCIA PROVÁVEL", "FALTA", "FALTA — CARGO NÃO ENCONTRADO"],
        default=["FALTA", "FALTA — CARGO NÃO ENCONTRADO"],
        key="epi_status_filter",
    )
    cargos_filter = st.multiselect(
        "Filtrar cargo — EPI",
        sorted(detail_df["Cargo PGR"].unique()),
        key="epi_role_filter",
    )
    view = detail_df.copy()
    if cargos_filter:
        view = view[view["Cargo PGR"].isin(cargos_filter)]
    if status_options:
        view = view[view["Status"].isin(status_options)]

    with st.expander("🔎 Pendências de EPI / conferência", expanded=True):
        st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("⚠️ Auditoria de Riscos — Anexo I")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Cargos com inventário", summary["cargos_inventario"])
r2.metric("Cargos sem Anexo I", summary["cargos_sem_inventario"])
r3.metric("Riscos atendidos", summary["riscos_atendidos"])
r4.metric("Riscos em falta", summary["riscos_em_falta"])
r5.metric("Riscos extras no sistema", summary["riscos_extras_sistema"])

if summary["riscos_em_falta"] == 0 and summary["cargos_sem_inventario"] == 0:
    st.success("Riscos: nenhum risco do Anexo I ficou sem correspondência no sistema.")
else:
    st.warning("Riscos: existem riscos do Anexo I sem correspondência no sistema ou cargos que não foram encontrados no inventário.")

if not risk_missing_df.empty:
    with st.expander("🚨 Riscos do Anexo I que NÃO aparecem no sistema", expanded=True):
        st.dataframe(risk_missing_df, use_container_width=True, hide_index=True)

with st.expander("➕ Riscos encontrados no sistema que NÃO aparecem no Anexo I"):
    if risk_extras_df.empty:
        st.write("Nenhum risco extra foi identificado.")
    else:
        st.dataframe(risk_extras_df, use_container_width=True, hide_index=True)

with st.expander("✅ Detalhamento das correspondências de riscos"):
    if risk_detail_df.empty:
        st.write("Nenhuma correspondência de risco foi gerada.")
    else:
        st.dataframe(risk_detail_df, use_container_width=True, hide_index=True)

with st.expander("📋 Cargos do sistema × Anexo I"):
    st.dataframe(risk_roles_df, use_container_width=True, hide_index=True)

with st.expander("⚠️ Cargos do sistema sem correspondência no Anexo I"):
    if risk_missing_roles_df.empty:
        st.write("Todos os cargos do sistema tiveram alguma correspondência automática no Anexo I.")
    else:
        st.dataframe(risk_missing_roles_df, use_container_width=True, hide_index=True)

with st.expander("🦺 Ver EPIs extras cadastrados no sistema"):
    if extras_df.empty:
        st.write("Nenhum EPI extra foi identificado.")
    else:
        st.dataframe(extras_df, use_container_width=True, hide_index=True)

with st.expander("📋 Ver cargos do Anexo IV sem correspondência no relatório"):
    if missing_roles_df.empty:
        st.write("Todos os cargos do Anexo IV tiveram alguma correspondência automática.")
    else:
        st.dataframe(missing_roles_df, use_container_width=True, hide_index=True)

excel_bytes = build_excel(
    summary,
    roles_df,
    detail_df,
    missing_epi_df,
    missing_roles_df,
    extras_df,
    risk_roles_df,
    risk_detail_df,
    risk_missing_df,
    risk_extras_df,
    risk_missing_roles_df,
)

st.download_button(
    "⬇️ Baixar auditoria completa em Excel",
    data=excel_bytes,
    file_name="auditoria_pgr_x_sistema_epi_riscos.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "A auditoria de riscos usa correspondência semântica conservadora: a família do risco precisa ser compatível "
    "e marcadores específicos (altura, espaço confinado, ruído, calor, químicos etc.) são preservados. "
    "Correspondências intermediárias aparecem como 'provável' para conferência humana."
)
