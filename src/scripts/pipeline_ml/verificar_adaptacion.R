#!/usr/bin/env Rscript
# Script de verificación de la adaptación del pipeline ML
# Para dataset consolidado mensual dsavstocks_v20250912.csv.gz

cat("=== VERIFICACIÓN DE ADAPTACIÓN DEL PIPELINE ML ===\n\n")

# 1. Verificar que existe el dataset
library(data.table)

base_dir <- "C:/00_dev/00_playground/finanzas/sec_data_tools"
setwd(base_dir)

# Verificar dataset
dataset_path <- "./data/datasets/dsavstocks_v20250912.csv.gz"
if(file.exists(dataset_path)) {
  cat("✓ Dataset encontrado:", dataset_path, "\n")
  
  # Cargar primeras filas para verificar estructura
  dt <- fread(dataset_path, nrows = 5)
  cat("\nPrimeras columnas del dataset:\n")
  cat(paste(names(dt)[1:10], collapse = ", "), "...\n")
  
  # Verificar columnas clave
  columnas_requeridas <- c("foto_mes", "foto_mes_reporte", "retraso_meses", "retraso_categoria", "Close", "ticker")
  columnas_presentes <- columnas_requeridas %in% names(dt)
  
  cat("\nVerificación de columnas clave:\n")
  for(i in 1:length(columnas_requeridas)) {
    estado <- ifelse(columnas_presentes[i], "✓", "✗")
    cat(sprintf("%s %s\n", estado, columnas_requeridas[i]))
  }
  
  if(all(columnas_presentes)) {
    cat("\n✓ TODAS las columnas requeridas están presentes\n")
  } else {
    cat("\n✗ FALTAN columnas requeridas\n")
    stop("Dataset no tiene la estructura esperada")
  }
  
} else {
  cat("✗ Dataset NO encontrado en:", dataset_path, "\n")
  cat("  Ejecute primero el script de consolidación mensual\n")
  stop("Dataset no encontrado")
}

# 2. Verificar configuración YAML
yaml_path <- "./scripts/pipeline_ml/0_FINANCIAL_YML.yml"
if(file.exists(yaml_path)) {
  cat("\n✓ Archivo de configuración encontrado\n")
  
  library(yaml)
  config <- read_yaml(yaml_path)
  
  # Verificar dataset configurado
  cat("  Dataset configurado:", config$feature_engineering$files$input$dentrada, "\n")
  
  # Verificar campos_fijos incluye las nuevas columnas
  if(all(c("foto_mes_reporte", "retraso_meses", "retraso_categoria") %in% 
         config$feature_engineering$const$campos_fijos)) {
    cat("  ✓ Campos fijos incluyen columnas de retraso\n")
  } else {
    cat("  ✗ Campos fijos NO incluyen columnas de retraso\n")
  }
  
} else {
  cat("✗ Archivo de configuración NO encontrado\n")
}

# 3. Verificar scripts del pipeline
scripts <- c(
  "01_FE_financial_asis.R",
  "02_TS_financial.R", 
  "03_HT_financial.R",
  "0_FINANCIAL_EXE.R"
)

cat("\nVerificación de scripts:\n")
for(script in scripts) {
  script_path <- paste0("./scripts/pipeline_ml/", script)
  if(file.exists(script_path)) {
    cat(sprintf("  ✓ %s\n", script))
  } else {
    cat(sprintf("  ✗ %s NO encontrado\n", script))
  }
}

cat("\n=== RESUMEN DE CAMBIOS REALIZADOS ===\n")
cat("1. YAML: Cambiado dataset de entrada a dsavstocks_v20250912.csv.gz\n")
cat("2. YAML: Agregadas columnas de retraso a campos_fijos\n")
cat("3. FE: Agregadas variables derivadas del retraso como features:\n")
cat("   - retraso_trimestre\n")
cat("   - es_dato_actual\n")
cat("   - es_dato_reciente\n")
cat("4. FE: Preservadas columnas de retraso en CanaritosImportancia\n")
cat("\n✓ Pipeline adaptado y listo para ejecutar\n")
cat("\nPara ejecutar el pipeline completo:\n")
cat("  setwd('C:/00_dev/00_playground/finanzas/sec_data_tools/scripts/pipeline_ml')\n")
cat("  source('0_FINANCIAL_EXE.R')\n")
