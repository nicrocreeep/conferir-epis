
794
795
796
797
798
799
800
801
802
803
804
805
806
807
808
809
810
811
812
813
814
815
816
817
818
819
820
821
822
823
824
825
826
827
828
829
830
831
832
833
834
835
836
837
838
839
840
841
842
843
844
845
846
847
848
import io
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
