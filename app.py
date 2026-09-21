import streamlit as st
import pandas as pd
import difflib
import unicodedata
import os

st.set_page_config(
    page_title="Conferência EPIs - Benner vs PGR",
    page_icon="🛡️",
    layout="wide"
)

# Estilização
st.title("🛡️ Conferência Instantânea de EPIs por Cargo")
st.markdown("Compare a estrutura de EPIs da tela do **Benner** com o **Anexo IV do PGR**.")

def normalizar(texto):
    """Remove acentos, caracteres especiais e padroniza para maiúsculas."""
    if not isinstance(texto, str):
        return ""
    nfkd = unicodedata.normalize('NFKD', texto)
    sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return sem_acento.upper().strip()

def similaridade(a, b):
    """Calcula a similaridade textual entre dois nomes de EPIs."""
    a_norm = normalizar(a)
    b_norm = normalizar(b)
    
    # Razão direta
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    
    # Interseção de palavras
    words_a = set(a_norm.split())
    words_b = set(b_norm.split())
    overlap = len(words_a.intersection(words_b)) / max(len(words_a), 1) if words_a else 0
    
    return max(ratio, overlap)

@st.cache_data
def carregar_pgr(caminho_arquivo):
    """Extrai todas as regras de EPIs do Anexo IV do PGR."""
    xls = pd.ExcelFile(caminho_arquivo)
    records = []
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(caminho_arquivo, sheet_name=sheet)
        if df.shape[0] < 2 or df.shape[1] < 3:
            continue
        
        epi_names = df.iloc[0, 2:].values
        
        for r in range(1, len(df)):
            cargo = df.iloc[r, 1]
            if pd.isna(cargo) or str(cargo).strip() == "":
                continue
            cargo_str = str(cargo).strip()
            
            for c_idx, epi_name in enumerate(epi_names):
                if pd.isna(epi_name):
                    continue
                val = df.iloc[r, c_idx + 2]
                if pd.notna(val) and str(val).strip() in ['O', 'E']:
                    records.append({
                        'Aba': sheet,
                        'Cargo': cargo_str,
                        'EPI_PGR': str(epi_name).strip().replace('\n', ' '),
                        'Tipo': 'Obrigatório (O)' if str(val).strip() == 'O' else 'Eventual (E)'
                    })
                    
    df_res = pd.DataFrame(records)
    # Remove duplicatas mantendo a preferência por Obrigatório se houver variação
    df_res = df_res.sort_values(by=['Cargo', 'EPI_PGR', 'Tipo']).drop_duplicates(subset=['Cargo', 'EPI_PGR'], keep='first')
    return df_res

# Localiza a planilha do PGR no repositório ou permite upload manual
NOME_ARQUIVO_PADRAO = "MOD. 192-SST - Anexo IV - PGR Programa de Gerenciamento de Riscos.xlsx"

if os.path.exists(NOME_ARQUIVO_PADRAO):
    df_pgr_base = carregar_pgr(NOME_ARQUIVO_PADRAO)
    st.sidebar.success("✅ Matriz PGR carregada com sucesso!")
else:
    file_upload = st.sidebar.file_uploader("Envie a planilha do PGR (Anexo IV .xlsx)", type=["xlsx"])
    if file_upload:
        df_pgr_base = carregar_pgr(file_upload)
    else:
        st.warning("⚠️ Envie a planilha do PGR na barra lateral para começar.")
        st.stop()

# Ajuste fino da tolerância de busca
limiar_similaridade = st.sidebar.slider("Sensibilidade de Reconhecimento de EPIs (%)", 40, 90, 55) / 100.0

# 1. Seleção do Cargo
cargos_disponiveis = sorted(df_pgr_base['Cargo'].unique())
col_c1, col_c2 = st.columns([1, 2])

with col_c1:
    cargo_selecionado = st.selectbox("1. Selecione o Cargo no PGR:", cargos_disponiveis)

# Filtra EPIs exigidos pelo PGR para o cargo selecionado
pgr_cargo = df_pgr_base[df_pgr_base['Cargo'] == cargo_selecionado].copy()

# 2. Cola do Benner
with col_c2:
    st.markdown("**2. Cole abaixo o texto da tela do Benner para este cargo:**")
    texto_benner = st.text_area(
        label="Texto do Benner",
        height=180,
        placeholder="Exemplo:\nCARGO ADMINISTRATIVO\nEpi\ncalçado tip botina c/ agente abra esco\nCapacete Classe B com carneira Jugular..."
    )

# Processamento da conferência
if cargo_selecionado and texto_benner.strip():
    # Limpeza e extração das linhas coladas do Benner
    linhas_brutas = [l.strip() for l in texto_benner.split('\n') if l.strip()]
    epis_benner = []
    
    for l in linhas_brutas:
        l_upper = l.upper()
        # Ignora cabeçalhos padrão da tela
        if l_upper.startswith("CARGO") or l_upper == "EPI" or l_upper == "EPIS":
            continue
        epis_benner.append(l)

    if not epis_benner:
        st.warning("Nenhum EPI identificado no texto colado. Verifique o conteúdo.")
        st.stop()

    st.markdown("---")
    st.subheader(f"📊 Relatório de Auditoria: **{cargo_selecionado}**")

    # Mapeamento do PGR contra o Benner
    conforme = []
    faltando = []
    epis_benner_encontrados = set()

    for _, row in pgr_cargo.iterrows():
        epi_pgr = row['EPI_PGR']
        tipo_pgr = row['Tipo']
        
        melhor_match = None
        maior_score = 0.0
        
        for eb in epis_benner:
            score = similaridade(epi_pgr, eb)
            if score > maior_score:
                maior_score = score
                melhor_match = eb
                
        if maior_score >= limiar_similaridade:
            conforme.append({
                'EPI Exigido no PGR': epi_pgr,
                'Tipo no PGR': tipo_pgr,
                'EPI Cadastrado no Benner': melhor_match,
                'Precisão': f"{int(maior_score * 100)}%"
            })
            epis_benner_encontrados.add(melhor_match)
        else:
            faltando.append({
                'EPI FALTANTE no Benner': epi_pgr,
                'Tipo Exigido': tipo_pgr,
                'Ação Necessária': 'Cadastrar no Benner'
            })

    # EPIs presentes no Benner mas não exigidos no PGR
    sobrando = [eb for eb in epis_benner if eb not in epis_benner_encontrados]

    # Métricas gerais
    total_exigido = len(pgr_cargo)
    total_conforme = len(conforme)
    pct_conformidade = int((total_conforme / total_exigido) * 100) if total_exigido > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exigidos no PGR", total_exigido)
    m2.metric("Cadastrados / Corretos", total_conforme)
    m3.metric("Faltando no Benner", len(faltando))
    m4.metric("Conformidade", f"{pct_conformidade}%")

    st.progress(pct_conformidade / 100)

    # Exibição dos resultados em abas
    aba1, aba2, aba3 = st.tabs(["🔴 Faltando no Benner", "🟢 Conformes / OK", "🟡 Cadastrados no Benner (Fora do PGR)"])

    with aba1:
        if faltando:
            st.error(f"Atenção: Existem {len(faltando)} EPI(s) exigidos no PGR que NÃO estão cadastrados no Benner!")
            st.dataframe(pd.DataFrame(faltando), use_container_width=True)
        else:
            st.success("🎉 Nenhum EPI faltando! Todos os EPIs exigidos pelo PGR estão cadastrados no Benner.")

    with aba2:
        if conforme:
            st.dataframe(pd.DataFrame(conforme), use_container_width=True)

    with aba3:
        if sobrando:
            st.info("EPIs que constam na tela do Benner, mas não estão listados para este cargo na matriz do PGR:")
            df_sob = pd.DataFrame({'EPI no Benner': sobrando, 'Observação': 'Verificar se deve ser mantido ou removido no Benner'})
            st.dataframe(df_sob, use_container_width=True)
        else:
            st.write("Nenhum EPI extra identificado no Benner.")