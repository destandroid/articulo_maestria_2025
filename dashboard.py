# dashboard.py
import pandas as pd
import numpy as np
import streamlit as st
import re

# ------------------- CONFIG -------------------
st.set_page_config(page_title="ENADE Dashboard", page_icon="📊", layout="wide")

# ----------------- UTILIDADES (Módulo 1) -----------------
def total_alunos_por_curso_from_df(df_in: pd.DataFrame) -> pd.DataFrame:
    necesarios = {'NOME_CURSO', 'NT_GER'}
    faltan = necesarios - set(df_in.columns)
    if faltan:
        raise KeyError(f"Faltan columnas en el CSV: {faltan}")

    df = df_in[['NOME_CURSO', 'NT_GER']].copy()

    df['CURSO_ORIGINAL'] = df['NOME_CURSO']
    df['NOME_CURSO'] = df['NOME_CURSO'].astype(str).str.strip().str.lower()

    contagem_total = df.groupby('NOME_CURSO', dropna=False).size()

    mask_valid = df['NT_GER'].astype(str).ne('') & df['NT_GER'].notna()
    contagem_validos = df[mask_valid].groupby('NOME_CURSO', dropna=False).size()
    contagem_validos = contagem_validos.reindex(contagem_total.index, fill_value=0)

    percentual_realizacao = (contagem_validos.div(contagem_total).mul(100)).round(2)

    result = pd.DataFrame({
        'Número total de alunos\ninscritos': contagem_total.astype(int),
        'Número de alunos\nque realizaram a prova': contagem_validos.astype(int),
        'Percentual que realizou\na prova (%)': percentual_realizacao
    }).reset_index()

    nomes_orig = (df[['NOME_CURSO', 'CURSO_ORIGINAL']]
                  .drop_duplicates(subset=['NOME_CURSO'])
                  .set_index('NOME_CURSO')['CURSO_ORIGINAL'])
    result['Cursos participantes do ENADE'] = result['NOME_CURSO'].map(nomes_orig)
    result.drop(columns=['NOME_CURSO'], inplace=True)

    soma_total_insc = result['Número total de alunos\ninscritos'].sum()
    soma_total_real = result['Número de alunos\nque realizaram a prova'].sum()
    perc_total = (soma_total_real / soma_total_insc * 100) if soma_total_insc else 0.0

    total_row = pd.DataFrame([{
        'Cursos participantes do ENADE': 'Total alunos ENADE',
        'Número total de alunos\ninscritos': soma_total_insc,
        'Número de alunos\nque realizaram a prova': soma_total_real,
        'Percentual que realizou\na prova (%)': round(perc_total, 2)
    }])

    result = pd.concat([result, total_row], ignore_index=True)

    cols_final = [
        'Cursos participantes do ENADE',
        'Número total de alunos\ninscritos',
        'Número de alunos\nque realizaram a prova',
        'Percentual que realizou\na prova (%)'
    ]
    return result[cols_final]

# ----------------- UTILIDADES (Módulo 2) -----------------
def notas_medianas_por_curso_from_df(df_in: pd.DataFrame) -> pd.DataFrame:
    necesarios = {'NOME_CURSO', 'NT_FG', 'NT_CE', 'NT_GER'}
    faltan = necesarios - set(df_in.columns)
    if faltan:
        raise KeyError(f"Faltan columnas en el CSV: {faltan}")

    df = df_in[['NOME_CURSO', 'NT_FG', 'NT_CE', 'NT_GER']].copy()

    df['CURSO_ORIGINAL'] = df['NOME_CURSO']
    df['NOME_CURSO'] = df['NOME_CURSO'].astype(str).str.strip().str.lower()

    for col in ['NT_FG', 'NT_CE', 'NT_GER']:
        s = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(s, errors='coerce')

    contagem_validos = df[df['NT_GER'].notna()].groupby('NOME_CURSO').size()
    medianas = df.groupby('NOME_CURSO')[['NT_FG', 'NT_CE', 'NT_GER']].median().round(2)

    result = contagem_validos.to_frame(name='Número de alunos\nque realizaram a prova') \
                             .join(medianas) \
                             .reset_index()

    nomes_orig = (df[['NOME_CURSO', 'CURSO_ORIGINAL']]
                  .drop_duplicates(subset=['NOME_CURSO'])
                  .set_index('NOME_CURSO')['CURSO_ORIGINAL'])
    result['Cursos participantes do ENADE'] = result['NOME_CURSO'].map(nomes_orig)
    result.drop(columns=['NOME_CURSO'], inplace=True)

    result = result[['Cursos participantes do ENADE',
                     'Número de alunos\nque realizaram a prova',
                     'NT_FG', 'NT_CE', 'NT_GER']]
    result['Número de alunos\nque realizaram a prova'] = result['Número de alunos\nque realizaram a prova'].astype(int)

    return result

# ----------------- UTILIDADES (Módulo 3) -----------------
def estudantes_por_faculdade_from_df(
    df_in: pd.DataFrame,
    cursos_analisados: list[str],
    faculdades_analisadas_CO_IES: list[str] | list[int],
    faculdade_destacada: str | int | None
) -> dict[str, pd.DataFrame]:
    """
    Para cada curso seleccionado:
      - Cuenta inscritos totales por IES
      - Cuenta quienes realizaron la prueba (NT_GER válido) por IES
      - Devuelve DataFrame con columnas: CO_IES, NOME_IES, Total inscritos, Realizaram, Destacada(⭐)
    """
    necesarios = {'NOME_CURSO', 'NOME_IES', 'CO_IES', 'NT_GER'}
    faltan = necesarios - set(df_in.columns)
    if faltan:
        raise KeyError(f"Faltan columnas en el CSV: {faltan}")

    df = df_in[['NOME_CURSO', 'NOME_IES', 'CO_IES', 'NT_GER']].copy()
    df['NOME_CURSO'] = df['NOME_CURSO'].astype(str).str.strip().str.lower()

    # Normaliza CO_IES a string para que los filtros funcionen bien
    df['CO_IES'] = df['CO_IES'].astype(str)
    faculdades_analisadas_CO_IES = [str(x) for x in faculdades_analisadas_CO_IES] if faculdades_analisadas_CO_IES else []

    # Filtro base por IES si hay selección
    if faculdades_analisadas_CO_IES:
        df = df[df['CO_IES'].isin(faculdades_analisadas_CO_IES)]

    # Conteo de válidos
    mask_valid = df['NT_GER'].astype(str).ne('') & df['NT_GER'].notna()
    resultados: dict[str, pd.DataFrame] = {}

    for curso in cursos_analisados:
        c_norm = str(curso).strip().lower()
        df_curso = df[df['NOME_CURSO'] == c_norm].copy()
        if df_curso.empty:
            resultados[curso] = pd.DataFrame(columns=[
                'CO_IES', 'NOME_IES', 'Número total de alunos\ninscritos',
                'Número de alunos\nque realizaram a prova', 'Destacada'
            ])
            continue

        total = (df_curso
                 .groupby(['CO_IES', 'NOME_IES'])
                 .size()
                 .rename('Número total de alunos\ninscritos'))
        validos = (df_curso[mask_valid]
                   .groupby(['CO_IES', 'NOME_IES'])
                   .size()
                   .rename('Número de alunos\nque realizaram a prova'))

        out = (pd.concat([total, validos], axis=1)
               .fillna(0)
               .astype({'Número total de alunos\ninscritos': int,
                        'Número de alunos\nque realizaram a prova': int})
               .reset_index()
               .sort_values(['Número total de alunos\ninscritos', 'Número de alunos\nque realizaram a prova'],
                            ascending=False))

        # Marcar destacada con ⭐ (si coincide por CO_IES)
        if faculdade_destacada is not None:
            fd = str(faculdade_destacada)
            out['Destacada'] = np.where(out['CO_IES'] == fd, '⭐', '')
        else:
            out['Destacada'] = ''

        resultados[curso] = out[['Destacada', 'CO_IES', 'NOME_IES',
                                 'Número total de alunos\ninscritos',
                                 'Número de alunos\nque realizaram a prova']]

    return resultados

# ----------------- CARGA DE DATOS -----------------
DIR_ENTRADA = 'microdados_tratados/form_microdados_arq3_vetores_filtrados.csv'

@st.cache_data(show_spinner=False)
def cargar_csv_fijo() -> pd.DataFrame:
    df = pd.read_csv(DIR_ENTRADA, sep=';')
    return df

# -------------------- SIDEBAR ---------------------
with st.sidebar:
    st.header("📚 Secciones")
    seccion = st.radio(
        "Selecciona una sección:",
        [
            "1) Cursos — Total de alumnos",
            "2) Notas — Medianas por curso",
            "3) Estudantes por faculdade (filtros)",
            "4) Desempeño por questão (percentuais por IES)",
            "5) Ranking por questão (Top N + destaque)",
            "6) Comparativo por curso (percentil da IES destacada)",


        ],
        index=0
    )
    st.caption("Usa el menú para navegar entre módulos.")

# ------------------- CONTENIDO --------------------
if seccion == "1) Cursos — Total de alumnos":
    st.title("📊 ENADE — Total de alumnos por curso")

    try:
        df_raw = cargar_csv_fijo()
    except Exception as e:
        st.error(f"No se pudo leer el archivo fijo en {DIR_ENTRADA}. Detalle: {e}")
        st.stop()

    total_registros = len(df_raw)
    total_cursos = df_raw['NOME_CURSO'].nunique() if 'NOME_CURSO' in df_raw.columns else 0
    c1, c2 = st.columns(2)
    c1.metric("Registros en CSV", f"{total_registros:,}".replace(",", "."))
    c2.metric("Cursos distintos", f"{total_cursos:,}".replace(",", "."))

    try:
        tabla = total_alunos_por_curso_from_df(df_raw)
    except KeyError as e:
        st.error(str(e))
        st.stop()

    st.subheader("📋 Tabla — Total de alumnos por curso")
    filtro = st.text_input("Filtrar por curso (contiene, sin distinguir mayúsculas/minúsculas):", value="")
    if filtro:
        mask = tabla['Cursos participantes do ENADE'].astype(str).str.lower().str.contains(filtro.strip().lower(), na=False)
        tabla_filtrada = tabla[mask].copy()
    else:
        tabla_filtrada = tabla.copy()

    st.dataframe(
        tabla_filtrada,
        use_container_width=True,
        hide_index=True
    )

    fila_total = tabla[tabla['Cursos participantes do ENADE'] == 'Total alunos ENADE']
    if not fila_total.empty:
        tot_insc = int(fila_total['Número total de alunos\ninscritos'].values[0])
        tot_real = int(fila_total['Número de alunos\nque realizaram a prova'].values[0])
        perc_tot = float(fila_total['Percentual que realizou\na prova (%)'].values[0])
        st.markdown("### 📌 Resumen global")
        c3, c4, c5 = st.columns(3)
        c3.metric("Total inscritos", f"{tot_insc:,}".replace(",", "."))
        c4.metric("Total que realizaron", f"{tot_real:,}".replace(",", "."))
        c5.metric("% realizó", f"{perc_tot:.2f}%")

elif seccion == "2) Notas — Medianas por curso":
    st.title("📊 ENADE — Notas (medianas) por curso")

    try:
        df_raw = cargar_csv_fijo()
    except Exception as e:
        st.error(f"No se pudo leer el archivo fijo en {DIR_ENTRADA}. Detalle: {e}")
        st.stop()

    try:
        tabla = notas_medianas_por_curso_from_df(df_raw)
    except KeyError as e:
        st.error(str(e))
        st.stop()

    st.subheader("📋 Tabla — Medianas de NT_FG, NT_CE, NT_GER por curso")
    filtro = st.text_input("Filtrar por curso (contiene, sin distinguir mayúsculas/minúsculas):", value="")
    if filtro:
        mask = tabla['Cursos participantes do ENADE'].astype(str).str.lower().str.contains(filtro.strip().lower(), na=False)
        tabla_filtrada = tabla[mask].copy()
    else:
        tabla_filtrada = tabla.copy()

    st.dataframe(
        tabla_filtrada,
        use_container_width=True,
        hide_index=True
    )

elif seccion == "3) Estudantes por faculdade (filtros)":
    st.title("🏫 ENADE — Estudantes por faculdade (por curso e IES)")

    try:
        df_raw = cargar_csv_fijo()
    except Exception as e:
        st.error(f"No se pudo leer el archivo fijo en {DIR_ENTRADA}. Detalle: {e}")
        st.stop()

    # Preparar listas para filtros
    cursos_all = sorted(df_raw['NOME_CURSO'].astype(str).str.strip().str.lower().unique()) if 'NOME_CURSO' in df_raw.columns else []
    ies_unique = df_raw[['CO_IES', 'NOME_IES']].dropna().copy() if {'CO_IES','NOME_IES'}.issubset(df_raw.columns) else pd.DataFrame(columns=['CO_IES','NOME_IES'])
    ies_unique['CO_IES'] = ies_unique['CO_IES'].astype(str)
    # Diccionario para mostrar "CO_IES — NOME_IES"
    ies_display = {row['CO_IES']: f"{row['CO_IES']} — {row['NOME_IES']}" for _, row in ies_unique.drop_duplicates('CO_IES').iterrows()}

    st.subheader("🎛️ Filtros")

    col_a, col_b = st.columns(2)
    with col_a:
        cursos_sel = st.multiselect(
            "Cursos analisados",
            options=cursos_all,
            default=cursos_all[:3] if len(cursos_all) >= 3 else cursos_all
        )
    with col_b:
        faculdades_sel = st.multiselect(
            "Faculdades analisadas (CO_IES)",
            options=list(ies_display.keys()),
            default=list(ies_display.keys())[:5],
            format_func=lambda k: ies_display.get(k, k)
        )

    # Faculdade destacada (opcional)
    faculdade_destacada = st.selectbox(
        "Faculdade destacada (opcional)",
        options=["(ninguna)"] + faculdades_sel if faculdades_sel else ["(ninguna)"],
        format_func=lambda k: "(ninguna)" if k == "(ninguna)" else ies_display.get(k, k),
        index=0
    )
    faculdade_destacada_val = None if faculdade_destacada == "(ninguna)" else faculdade_destacada

    if not cursos_sel:
        st.info("Selecciona al menos un curso.")
        st.stop()
    if not faculdades_sel:
        st.info("Selecciona al menos una faculdade (CO_IES).")
        st.stop()

    # Cálculo
    try:
        resultados = estudantes_por_faculdade_from_df(
            df_raw,
            cursos_analisados=cursos_sel,
            faculdades_analisadas_CO_IES=faculdades_sel,
            faculdade_destacada=faculdade_destacada_val
        )
    except KeyError as e:
        st.error(str(e))
        st.stop()

    # Mostrar una tabla por curso
    for curso in cursos_sel:
        st.markdown(f"### 📋 {curso.title()}")
        df_curso = resultados.get(curso, pd.DataFrame())
        if df_curso.empty:
            st.warning("Sin registros para los filtros seleccionados.")
            continue

        # Colocar IES destacada (⭐) arriba
        # --- ordenar poniendo la IES destacada arriba, pero SIN columna extra ---
        df_curso['_is_dest'] = (df_curso['CO_IES'].astype(str) == str(faculdade_destacada_val)) if faculdade_destacada_val else False
        df_curso = (df_curso
                    .sort_values(['_is_dest',
                                'Número total de alunos\ninscritos',
                                'Número de alunos\nque realizaram a prova'],
                                ascending=[False, False, False])
                    .drop(columns=['Destacada', '_is_dest'], errors='ignore')
                    .reset_index(drop=True)
                )

        # --- estilo para pintar de verde la fila destacada ---
        def _highlight_destacada(row):
            if faculdade_destacada_val and str(row['CO_IES']) == str(faculdade_destacada_val):
                return ['background-color: #9FFD9F'] * len(row)  # verde suave
            return [''] * len(row)

        st.dataframe(
            df_curso.style.apply(_highlight_destacada, axis=1),
            use_container_width=True,
            hide_index=True
        )

elif seccion == "4) Desempeño por questão (percentuais por IES)":
    st.title("📈 ENADE — Percentuais de acertos por IES e por questão")

    # Carga fija
    try:
        df_raw = cargar_csv_fijo()
    except Exception as e:
        st.error(f"No se pudo leer el archivo fijo en {DIR_ENTRADA}. Detalle: {e}")
        st.stop()

    # ------ Fuentes para filtros ------
    cursos_all = sorted(df_raw['NOME_CURSO'].astype(str).str.strip().str.lower().unique()) if 'NOME_CURSO' in df_raw.columns else []
    ies_unique = df_raw[['CO_IES', 'NOME_IES']].dropna().copy() if {'CO_IES','NOME_IES'}.issubset(df_raw.columns) else pd.DataFrame(columns=['CO_IES','NOME_IES'])
    ies_unique['CO_IES'] = ies_unique['CO_IES'].astype(str)
    ies_display = {row['CO_IES']: f"{row['CO_IES']} — {row['NOME_IES']}" for _, row in ies_unique.drop_duplicates('CO_IES').iterrows()}

    # Detectar columnas de questões (ajusta prefijos si hace falta)
    prefixes = (
        "forma_geral_q_", "forma_espec_q_",
        "forma_geral_alternativa_q_", "forma_espec_alternativa_q_"
    )
    cols_questoes = [c for c in df_raw.columns if any(c.startswith(p) for p in prefixes)]

    st.subheader("🎛️ Filtros")

    col1, col2 = st.columns(2)
    with col1:
        cursos_sel = st.multiselect(
            "Cursos analisados",
            options=cursos_all,
            default=cursos_all[:3] if len(cursos_all) >= 3 else cursos_all
        )
    with col2:
        faculdades_sel = st.multiselect(
            "Faculdades analisadas (CO_IES)",
            options=list(ies_display.keys()),
            default=list(ies_display.keys())[:5],
            format_func=lambda k: ies_display.get(k, k)
        )

    questoes_sel = st.multiselect(
        "Questões (colunas) para calcular percentuais",
        options=cols_questoes,
        default=cols_questoes[:10] if len(cols_questoes) > 10 else cols_questoes
    )

    faculdade_destacada = st.selectbox(
        "Faculdade destacada (opcional)",
        options=["(ninguna)"] + faculdades_sel if faculdades_sel else ["(ninguna)"],
        format_func=lambda k: "(ninguna)" if k == "(ninguna)" else ies_display.get(k, k),
        index=0
    )
    faculdade_destacada_val = None if faculdade_destacada == "(ninguna)" else faculdade_destacada

    if not cursos_sel:
        st.info("Selecciona al menos un curso.")
        st.stop()
    if not faculdades_sel:
        st.info("Selecciona al menos una faculdade (CO_IES).")
        st.stop()
    if not questoes_sel:
        st.info("Selecciona al menos una questão (columna).")
        st.stop()

    # ------ Base filtrada por IES (luego se filtra por curso en el loop) ------
    df_base = df_raw[['CO_IES', 'NOME_IES', 'NOME_CURSO'] + questoes_sel].copy()
    df_base['NOME_CURSO'] = df_base['NOME_CURSO'].astype(str).str.strip().str.lower()
    df_base['CO_IES'] = df_base['CO_IES'].astype(str)
    df_base = df_base[df_base['CO_IES'].isin(faculdades_sel)]

    # Mapeo CO_IES -> Nombre (sobre el subset de IES seleccionadas)
    mapeamento_ies_nome = (df_base[['CO_IES','NOME_IES']]
                           .drop_duplicates('CO_IES')
                           .set_index('CO_IES')['NOME_IES'])

    # ---------- BLOQUES POR CURSO ----------
    for curso in cursos_sel:
        st.divider()
        st.subheader(f"📋 {curso.title()} — Percentuais de acertos por IES (média por questão)")

        df_curso = df_base[df_base['NOME_CURSO'] == curso].copy()
        if df_curso.empty:
            st.warning("Sin datos para este curso con los filtros seleccionados.")
            continue

        # Asegurar numéricos y calcular percentuais *100
        for col in questoes_sel:
            s = df_curso[col].astype(str).str.replace(',', '.', regex=False)
            df_curso[col] = pd.to_numeric(s, errors='coerce')

        percentuais = df_curso.groupby('CO_IES')[questoes_sel].mean() * 100
        if percentuais.empty:
            st.warning("No hay percentuais calculables para este curso.")
            continue

        # ---- Tabla de percentuais por IES (redondeado a 2 decimales) ----
        # ---- Tabla de percentuais por IES (redondeado a 2 decimales) ----
        percentuais_show = percentuais.round(2).copy()
        percentuais_show.insert(
            0,
            'Nome da Instituição',
            percentuais_show.index.map(mapeamento_ies_nome).fillna('Desconhecido')
        )
        percentuais_show = percentuais_show.reset_index().rename(columns={'CO_IES': 'CO_IES (chave)'})

        # Resaltar fila de IES destacada en verde
        def _hl_percent(row):
            if faculdade_destacada_val and str(row['CO_IES (chave)']) == str(faculdade_destacada_val):
                return ['background-color: #A4FFA4'] * len(row)
            return [''] * len(row)

        # Formatear cada celda de questão: >100 -> "ANULADO", si no -> 2 decimales
        def _fmt_cell(x):
            import math
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return ""
            try:
                xv = float(x)
                return "ANULADO" if xv > 100 else f"{xv:.2f}"
            except Exception:
                return x

        # Pintar de rojo solo celdas >100 (sin tocar el resto del estilo)
        def _red_if_over_100(x):
            try:
                return 'background-color: #FFA4A4' if (x is not None and float(x) > 100) else ''
            except Exception:
                return ''

        # Columnas de questões presentes en la tabla
        quest_cols_presentes = [c for c in questoes_sel if c in percentuais_show.columns]

        st.dataframe(
            percentuais_show
                .style
                .format({c: _fmt_cell for c in quest_cols_presentes})
                .apply(_hl_percent, axis=1)                    # fila verde (IES destacada)
                .applymap(_red_if_over_100, subset=quest_cols_presentes),  # celda roja si >100
            use_container_width=True,
            hide_index=True
        )

       


        # ---------- Mejor y peor IES por questão (dos tablas separadas) ----------
        st.subheader("🏅 Melhor e pior IES por questão (dentro do filtro)")

        # ---- helper: resaltar SOLO si el CO_IES de la celda coincide exactamente ----
        def _hl_mp_exact(row):
            m = re.search(r"\((\d+)\)\s*$", str(row.get("IES", "")))  # extrae el CO_IES al final: ... (1234)
            co = m.group(1) if m else None
            if faculdade_destacada_val and co == str(faculdade_destacada_val):
                return ["background-color: #A4FFA4"] * len(row)
            return [""] * len(row)

        # ---------- MEJORES ----------
        melhores_rows = []
        for q in questoes_sel:
            s = percentuais[q].dropna()
            if s.empty:
                melhores_rows.append({"Questão": q, "IES": "—", "Percentual (%)": np.nan})
            else:
                co = s.idxmax()
                nome = mapeamento_ies_nome.get(co, co)
                val = float(s.loc[co])
                melhores_rows.append({"Questão": q, "IES": f"{nome} ({co})", "Percentual (%)": round(val, 2)})

        df_melhores = pd.DataFrame(melhores_rows)

        # ---------- PEORES ----------
        piores_rows = []
        for q in questoes_sel:
            s = percentuais[q].dropna()
            if s.empty:
                piores_rows.append({"Questão": q, "IES": "—", "Percentual (%)": np.nan})
            else:
                co = s.idxmin()
                nome = mapeamento_ies_nome.get(co, co)
                val = float(s.loc[co])
                piores_rows.append({"Questão": q, "IES": f"{nome} ({co})", "Percentual (%)": round(val, 2)})

        df_piores = pd.DataFrame(piores_rows)

        # ---- mostrar en dos tablas, con formato a 2 decimales y highlight exacto por CO_IES ----
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Melhor IES por questão**")
            st.dataframe(
                df_melhores.style.format({"Percentual (%)": "{:.2f}"}).apply(_hl_mp_exact, axis=1),
                use_container_width=True,
                hide_index=True
            )
        with colB:
            st.markdown("**Pior IES por questão**")
            st.dataframe(
                df_piores.style.format({"Percentual (%)": "{:.2f}"}).apply(_hl_mp_exact, axis=1),
                use_container_width=True,
                hide_index=True
            )

elif seccion == "5) Ranking por questão (Top N + destaque)":
    st.title("🏆 ENADE — Ranking por questão (Top N + destaque)")

    # Carga fija
    try:
        df_raw = cargar_csv_fijo()
    except Exception as e:
        st.error(f"No se pudo leer el archivo fijo en {DIR_ENTRADA}. Detalle: {e}")
        st.stop()

    # --- Fuentes para filtros ---
    cursos_all = sorted(df_raw['NOME_CURSO'].astype(str).str.strip().str.lower().unique()) if 'NOME_CURSO' in df_raw.columns else []
    ies_unique = df_raw[['CO_IES', 'NOME_IES']].dropna().copy() if {'CO_IES','NOME_IES'}.issubset(df_raw.columns) else pd.DataFrame(columns=['CO_IES','NOME_IES'])
    ies_unique['CO_IES'] = ies_unique['CO_IES'].astype(str)
    ies_display = {row['CO_IES']: f"{row['CO_IES']} — {row['NOME_IES']}" for _, row in ies_unique.drop_duplicates('CO_IES').iterrows()}

    # Detectar columnas de questões (ajusta si tus prefijos cambian)
    prefixes = ("forma_geral_q_", "forma_espec_q_", "forma_geral_alternativa_q_", "forma_espec_alternativa_q_")
    cols_questoes = [c for c in df_raw.columns if any(c.startswith(p) for p in prefixes)]

    st.subheader("🎛️ Filtros")
    col1, col2 = st.columns(2)
    with col1:
        cursos_sel = st.multiselect("Cursos analisados", options=cursos_all,
                                    default=cursos_all[:2] if len(cursos_all) >= 2 else cursos_all)
    with col2:
        top_n = st.slider("Top N por questão", min_value=1, max_value=20, value=5, step=1)

    questoes_sel = st.multiselect("Questões (colunas) para o ranking", options=cols_questoes,
                                  default=cols_questoes[:10] if len(cols_questoes) > 10 else cols_questoes)

    # Destaque (CO_IES) — opcional
    fac_destaque = st.selectbox("Faculdade (CO_IES) de destaque (opcional)",
                                options=["(ninguna)"] + list(ies_display.keys()),
                                format_func=lambda k: "(ninguna)" if k == "(ninguna)" else ies_display.get(k, k),
                                index=0)
    fac_destaque_val = None if fac_destaque == "(ninguna)" else str(fac_destaque)

    if not cursos_sel:
        st.info("Selecciona al menos un curso.")
        st.stop()
    if not questoes_sel:
        st.info("Selecciona al menos una questão.")
        st.stop()

    # --- Base común
    df_base = df_raw[['CO_IES', 'NOME_IES', 'NOME_CURSO'] + questoes_sel].copy()
    df_base['CO_IES'] = df_base['CO_IES'].astype(str)
    df_base['NOME_CURSO'] = df_base['NOME_CURSO'].astype(str).str.strip().str.lower()
    for col in questoes_sel:
        s = df_base[col].astype(str).str.replace(',', '.', regex=False)
        df_base[col] = pd.to_numeric(s, errors='coerce')

    # --- Loop por curso (cada curso es un bloque con divisor) ---
    for curso in cursos_sel:
        st.divider()
        st.subheader(f"📋 {curso.title()} — Ranking por questão")

        df_curso = df_base[df_base['NOME_CURSO'] == curso].copy()
        if df_curso.empty:
            st.warning("Sin datos para este curso.")
            continue

        # Mapeo CO_IES -> Nombre (en el subset del curso)
        map_ies_nome = (df_curso[['CO_IES', 'NOME_IES']]
                        .drop_duplicates('CO_IES')
                        .set_index('CO_IES')['NOME_IES'])

        # Percentuais por IES (media * 100) para cada questão
        percentuais = (df_curso.groupby('CO_IES')[questoes_sel].mean() * 100)

        # Construir ranking por questão: Top N + destaque (si no entra en el Top N)
        rows = []
        for q in questoes_sel:
            s = percentuais[q].dropna().sort_values(ascending=False)
            if s.empty:
                continue

            # Top N
            s_top = s.head(top_n)

            # Asegurar destaque (si existe en la série y no está en el Top)
            if fac_destaque_val and fac_destaque_val in s.index and fac_destaque_val not in s_top.index:
                s_top = pd.concat([s_top, s.loc[[fac_destaque_val]]])

            # Volcar filas (con rank real de la serie completa)
            for co_ies, val in s_top.items():
                pos = int(s.index.get_loc(co_ies)) + 1  # ranking 1-based
                rows.append({
                    "Questão": q,
                    "Posição": pos,
                    "CO_IES (chave)": co_ies,
                    "IES": f"{map_ies_nome.get(co_ies, co_ies)} ({co_ies})",
                    "Percentual (%)": round(float(val), 2)
                })

        if not rows:
            st.warning("No hay percentuais calculables en este curso para las questões seleccionadas.")
            continue

        df_rank = pd.DataFrame(rows).sort_values(["Questão", "Posição"], ascending=[True, True]).reset_index(drop=True)

        # Estilo: resaltar SOLO la fila cuyo CO_IES coincide exactamente con la faculdade destacada
        def _hl_rank(row):
            if fac_destaque_val and str(row["CO_IES (chave)"]) == fac_destaque_val:
                return ["background-color: #A4FFA4"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_rank.style.format({"Percentual (%)": "{:.2f}"}).apply(_hl_rank, axis=1),
            use_container_width=True,
            hide_index=True
        )


elif seccion == "6) Comparativo por curso (percentil da IES destacada)":
    st.title("🎯 ENADE — Comparativo por curso (percentil da IES destacada)")

    # Carga fija
    try:
        df_raw = cargar_csv_fijo()
    except Exception as e:
        st.error(f"No se pudo leer el archivo fijo en {DIR_ENTRADA}. Detalle: {e}")
        st.stop()

    # ---- Detectar columnas de questões (ajusta prefijos si difieren en tu dataset)
    prefixes = ("forma_geral_q_", "forma_espec_q_",
                "forma_geral_alternativa_q_", "forma_espec_alternativa_q_")
    cols_questoes = [c for c in df_raw.columns if any(c.startswith(p) for p in prefixes)]
    if not cols_questoes:
        st.warning("No se detectaron columnas de questões.")
        st.stop()

    # ---- Fuentes para filtros
    cursos_all = sorted(df_raw['NOME_CURSO'].astype(str).str.strip().str.lower().unique()) \
                 if 'NOME_CURSO' in df_raw.columns else []
    ies_unique = df_raw[['CO_IES', 'NOME_IES']].dropna().copy() \
                 if {'CO_IES', 'NOME_IES'}.issubset(df_raw.columns) \
                 else pd.DataFrame(columns=['CO_IES','NOME_IES'])
    ies_unique['CO_IES'] = ies_unique['CO_IES'].astype(str)
    ies_display = {row['CO_IES']: f"{row['CO_IES']} — {row['NOME_IES']}"
                   for _, row in ies_unique.drop_duplicates('CO_IES').iterrows()}

    st.subheader("🎛️ Filtros")
    col1, col2 = st.columns(2)
    with col1:
        cursos_sel = st.multiselect(
            "Curso(s)",
            options=cursos_all,
            default=cursos_all[:2] if len(cursos_all) >= 2 else cursos_all
        )
    with col2:
        faculdade_destacada = st.selectbox(
            "Faculdade destacada (CO_IES)",
            options=list(ies_display.keys()),
            format_func=lambda k: ies_display.get(k, k),
            index=0 if ies_display else None
        )
    if not cursos_sel:
        st.info("Selecciona al menos un curso.")
        st.stop()

    # ---- Base común (incluye NT_GER si existe para detectar "hizo la prueba")
    cols_base = ['CO_IES', 'NOME_IES', 'NOME_CURSO'] + cols_questoes + (['NT_GER'] if 'NT_GER' in df_raw.columns else [])
    df_base = df_raw[cols_base].copy()
    df_base['CO_IES'] = df_base['CO_IES'].astype(str)
    df_base['NOME_CURSO'] = df_base['NOME_CURSO'].astype(str).str.strip().str.lower()

    # Asegurar numéricos por si hubiera comas
    for col in cols_questoes:
        s = df_base[col].astype(str).str.replace(',', '.', regex=False)
        df_base[col] = pd.to_numeric(s, errors='coerce')

    # ---- Un bloque por curso
    for curso in cursos_sel:
        st.divider()
        st.subheader(f"📋 {curso.title()}")

        df_curso = df_base[df_base['NOME_CURSO'] == curso].copy()
        if df_curso.empty:
            st.warning("Sin datos para este curso.")
            continue

        # KPI: nombre IES destacada + cuántas IES tienen este curso y realizaron la prueba
        if 'NT_GER' in df_curso.columns:
            has_exam = df_curso['NT_GER'].notna()
        else:
            # Fallback: al menos una questão respondida
            has_exam = df_curso[cols_questoes].notna().any(axis=1)

        n_ies_examen = df_curso.loc[has_exam, 'CO_IES'].nunique()

        nome_destacada = df_curso.loc[df_curso['CO_IES'] == str(faculdade_destacada), 'NOME_IES'].dropna().unique()
        nome_destacada = nome_destacada[0] if len(nome_destacada) else ies_display.get(str(faculdade_destacada), str(faculdade_destacada))

        c1, c2 = st.columns(2)
        c1.metric("Faculdade destacada", nome_destacada)
        c2.metric("IES com este curso e prova", f"{n_ies_examen}")

        # Percentuais por IES (média por questão * 100)
        percentuais = (df_curso.groupby('CO_IES')[cols_questoes].mean() * 100)

        # Si la IES destacada no está en este curso, avisar
        if str(faculdade_destacada) not in percentuais.index:
            st.warning("A IES destacada não possui registros para este curso.")
            continue

        # Percentil vectorizado por questão (sin ordenar)
        # rank: 1 = melhor (maior percentual), N = pior
        ranks = percentuais.rank(method='min', ascending=False)
        totals = ranks.count()  # total de IES válidas por questão
        row_rank = ranks.loc[str(faculdade_destacada)]

        percentil = (totals - row_rank + 1) / totals * 100.0
        df_percentil = pd.DataFrame({
            "Questão": percentil.index,
            "Percentil da IES destacada (%)": percentil.values.round(2)
        })

        # Fila média
        media_val = pd.to_numeric(df_percentil["Percentil da IES destacada (%)"], errors='coerce').mean()
        df_media = pd.DataFrame([{
            "Questão": "Média",
            "Percentil da IES destacada (%)": round(float(media_val), 2) if pd.notna(media_val) else None
        }])
        df_percentil = pd.concat([df_percentil, df_media], ignore_index=True)

        # Mostrar tabla (dos decimales)
        st.dataframe(
            df_percentil.style.format({"Percentil da IES destacada (%)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )





else:
    st.title("🧩 Módulo desconocido")
    st.info("Elige una opción válida en el menú.")
