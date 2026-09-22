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


def build_excel(summary, roles_df, detail_df, missing_epi_df, missing_roles_df, extras_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Resumo", index=False)
        roles_df.to_excel(writer, sheet_name="Cargos", index=False)
        detail_df.to_excel(writer, sheet_name="Detalhado", index=False)
        missing_epi_df.to_excel(writer, sheet_name="Faltas EPI", index=False)
        missing_roles_df.to_excel(writer, sheet_name="Cargos ausentes", index=False)
        extras_df.to_excel(writer, sheet_name="Extras sistema", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#17365D",
            "font_color": "white",
            "border": 1,
        })
        ok_fmt = workbook.add_format({"bg_color": "#E2F0D9"})
        fail_fmt = workbook.add_format({"bg_color": "#FCE4D6"})

        # Usa os próprios DataFrames para definir o intervalo do filtro.
        # Não usamos ws.dim_rowmax/ws.dim_colmax porque o XlsxWriter pode
        # deixá-los como None em planilhas sem linhas/colunas de dados.
        sheet_frames = {
            "Resumo": pd.DataFrame([summary]),
            "Cargos": roles_df,
            "Detalhado": detail_df,
            "Faltas EPI": missing_epi_df,
            "Cargos ausentes": missing_roles_df,
            "Extras sistema": extras_df,
        }

        for sheet_name, df in sheet_frames.items():
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)

            if len(df.columns) > 0:
                # Linha 0 é o cabeçalho; a última linha de dados é len(df).
                last_row = max(0, len(df))
                last_col = len(df.columns) - 1
                ws.autofilter(0, 0, last_row, last_col)

        # Reaplica larguras de forma simples.
        for sheet_name, df in [
            ("Cargos", roles_df),
            ("Detalhado", detail_df),
            ("Faltas EPI", missing_epi_df),
            ("Cargos ausentes", missing_roles_df),
            ("Extras sistema", extras_df),
        ]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                width = min(max(len(str(col)) + 2, 14), 45)
                if not df.empty:
                    sample = df[col].astype(str).head(100)
                    width = min(max(width, int(sample.map(len).max()) + 2), 55)
                ws.set_column(idx, idx, width)
            for idx, col in enumerate(df.columns):
                ws.write(0, idx, col, header_fmt)

            if sheet_name == "Detalhado" and not df.empty:
                status_col = df.columns.get_loc("Status")
                for row_idx, val in enumerate(df["Status"].astype(str), start=1):
                    if val.startswith("OK"):
                        ws.write(row_idx, status_col, val, ok_fmt)
                    elif val.startswith("FALTA"):
                        ws.write(row_idx, status_col, val, fail_fmt)

    output.seek(0)
    return output.getvalue()


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------

st.title("🦺 Auditor de EPIs — Anexo IV do PGR × Sistema")
st.markdown(
    "O sistema verifica se **todo EPI marcado no Anexo IV do PGR aparece no relatório do sistema para o respectivo cargo**. "
    "EPIs adicionais no sistema não geram falta."
)

with st.sidebar:
    st.header("Arquivos")
    st.caption("Para esta versão, use preferencialmente o Anexo IV em Excel (.xlsx).")
    report_file = st.file_uploader(
        "1. Relatório do sistema",
        type=["xlsx", "xls"],
        key="report",
    )
    pgr_file = st.file_uploader(
        "2. Anexo IV do PGR",
        type=["xlsx", "xls"],
        key="pgr",
    )

    st.divider()
    st.caption("Critério: qualquer célula preenchida no cruzamento Cargo × EPI do PGR é tratada como EPI exigido. Os valores O e E são preservados no relatório, sem interpretar o significado deles.")

if not report_file or not pgr_file:
    st.info("Envie os dois arquivos na barra lateral para iniciar a análise.")
    st.stop()

try:
    report = read_report(report_file.getvalue())
    pgr, epi_names = read_pgr(pgr_file.getvalue())
except Exception as exc:
    st.error(f"Não foi possível ler os arquivos: {exc}")
    st.stop()

with st.spinner("Cruzando cargos e EPIs..."):
    summary, roles_df, detail_df, missing_epi_df, missing_roles_df, extras_df = analyze(
        report, pgr, epi_names
    )

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cargos no PGR", summary["cargos_pgr"])
m2.metric("Cargos atendidos", summary["cargos_atendidos"])
m3.metric("Cargos com falta", summary["cargos_com_falha"])
m4.metric("Cargos não encontrados", summary["cargos_nao_encontrados"])
m5.metric("EPIs em falta", summary["epis_em_falta"])

st.divider()

if summary["epis_em_falta"] == 0 and summary["cargos_nao_encontrados"] == 0:
    st.success("Nenhuma falta foi encontrada pelo critério automático da análise.")
else:
    st.warning("Existem cargos ou EPIs do PGR que não foram localizados automaticamente no relatório do sistema.")

# Filtros rápidos
if not detail_df.empty:
    cargos_filter = st.multiselect(
        "Filtrar cargo",
        sorted(detail_df["Cargo PGR"].unique()),
    )
    status_options = st.multiselect(
        "Filtrar status",
        ["OK", "OK — CORRESPONDÊNCIA PROVÁVEL", "FALTA", "FALTA — CARGO NÃO ENCONTRADO"],
        default=["FALTA", "FALTA — CARGO NÃO ENCONTRADO"],
    )
    view = detail_df.copy()
    if cargos_filter:
        view = view[view["Cargo PGR"].isin(cargos_filter)]
    if status_options:
        view = view[view["Status"].isin(status_options)]

    st.subheader("🔎 Pendências / conferência")
    st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("📋 Cargos do PGR")
st.dataframe(roles_df, use_container_width=True, hide_index=True)

with st.expander("Ver EPIs extras cadastrados no sistema"):
    if extras_df.empty:
        st.write("Nenhum EPI extra foi identificado.")
    else:
        st.dataframe(extras_df, use_container_width=True, hide_index=True)

with st.expander("Ver cargos do PGR sem correspondência no relatório"):
    if missing_roles_df.empty:
        st.write("Todos os cargos tiveram alguma correspondência automática.")
    else:
        st.dataframe(missing_roles_df, use_container_width=True, hide_index=True)

excel_bytes = build_excel(
    summary,
    roles_df,
    detail_df,
    missing_epi_df,
    missing_roles_df,
    extras_df,
)

st.download_button(
    "⬇️ Baixar auditoria em Excel",
    data=excel_bytes,
    file_name="auditoria_pgr_x_sistema_epi.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "Observação: correspondências com confiança intermediária são marcadas como 'provável' para conferência humana. "
    "Isso é especialmente útil quando o relatório usa abreviações ou nomes de cargo diferentes do PGR."
)
