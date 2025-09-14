# ENADE 2023 — Procesamiento y Dashboard Interactivo

Este repositorio contiene el código para procesar los **microdatos del ENADE 2023** y visualizar resultados de desempeño por curso e institución a través de un **dashboard interactivo** desarrollado en [Streamlit].

---

## 📂 Datos necesarios

Antes de ejecutar el procesamiento y el dashboard, descarga los siguientes archivos:

1. **Microdados ENADE 2023**  
   - Descargar desde: [INEP — Microdados ENADE 2023](https://download.inep.gov.br/microdados/microdados_enade_2023.zip)  
   - Descomprimir y colocar el contenido en la carpeta principal del proyecto.

2. **Mapeamento de IES**  
   - Descargar desde: [MEC — Dados Cursos Graduação Brasil](https://dadosabertos.mec.gov.br/images/conteudo/Ind-ensino-superior/2022//PDA_Dados_Cursos_Graduacao_Brasil.csv)  
   - Guardar el archivo en la carpeta `aux_files/`.

---

## ⚙️ Requisitos

- [Python 3.10+](https://www.python.org/)  
- Librerías principales:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `streamlit`

## 🚀Ejecución

- Procesamiento de datos: Abrir el cuaderno de Jupyter y ejecutar paso a paso el flujo de procesamiento:


 1_Preprocesamiento.ipynb

- Ejecutar el dashboard: Una vez procesados los datos y generados los archivos intermedios:
```console
streamlit run dashboard.py
```

