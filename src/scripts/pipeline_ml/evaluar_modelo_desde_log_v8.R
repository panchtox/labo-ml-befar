#!/usr/bin/env Rscript
# Script para evaluar modelo leyendo los mejores parámetros desde BO_log.txt
# Versión 8: Replicando exactamente la lógica del HT original

rm(list = ls())
gc()

require("data.table")
require("yaml")
require("lightgbm")
library(tidyverse)
library(ggrepel)

cat("=== EVALUACIÓN DE MODELO DESDE BO_log.txt (v8 - HT Match) ===\n\n")

# Función auxiliar para sumar meses en formato YYYYMM
sumar_meses <- function(fecha_yyyymm, meses) {
  # Convertir YYYYMM a fecha real
  año <- fecha_yyyymm %/% 100
  mes <- fecha_yyyymm %% 100
  
  # Sumar meses
  mes_nuevo <- mes + meses
  
  # Ajustar si nos pasamos de 12
  while(mes_nuevo > 12) {
    mes_nuevo <- mes_nuevo - 12
    año <- año + 1
  }
  
  # Reconstruir en formato YYYYMM
  return(año * 100 + mes_nuevo)
}

# Configuración del experimento
carpeta_base <- "C:/00_dev/00_playground/finanzas/sec_data_tools"
setwd(carpeta_base)

# Configurar experimento
experiment_label <- "avtk55resync"  # Ajustar según tu experimento
experiment_code <- "202409"          # Ajustar según tu experimento
orden_lead <- 1                     

experiment_dir <- paste(experiment_label, experiment_code, sep = "_")
experiment_lead_dir <- paste(experiment_label, experiment_code, paste0("f", orden_lead), sep = "_")

setwd("./exp")
setwd(experiment_dir)
setwd(experiment_lead_dir)

# Cargar parámetros
jsonfile <- list.files(pattern = ".json")[1]
PARAMS <- jsonlite::fromJSON(jsonfile)
cat("Experimento cargado:", experiment_lead_dir, "\n\n")

# Buscar y leer el BO_log.txt
if(dir.exists("03_HT")) {
  setwd("03_HT")
} else {
  stop("No existe directorio 03_HT")
}

if(!file.exists("BO_log.txt")) {
  stop("No se encontró BO_log.txt. Ejecute primero el HT.")
}

#------------------------------------------------------------------------------
# PARTE 1: Leer y procesar el log de optimización bayesiana
#------------------------------------------------------------------------------

cat("=== LEYENDO RESULTADOS DE OPTIMIZACIÓN BAYESIANA ===\n")

# Leer con fread directamente (como hace el HT con BOlog)
tabla_log <- fread("BO_log.txt")

if(nrow(tabla_log) == 0) {
  stop("BO_log.txt está vacío")
}

cat("Iteraciones encontradas:", nrow(tabla_log), "\n")

# Verificar columnas esperadas
columnas_requeridas <- c("learning_rate", "feature_fraction", "num_leaves", 
                         "min_data_in_leaf", "lambda_l1", "lambda_l2", 
                         "bagging_fraction", "ganancia")

if(!all(columnas_requeridas %in% names(tabla_log))) {
  columnas_faltantes <- setdiff(columnas_requeridas, names(tabla_log))
  stop(paste("Faltan columnas en BO_log.txt:", paste(columnas_faltantes, collapse=", ")))
}

# Encontrar el mejor resultado (como hace el HT)
mejor_iteracion <- tabla_log[which.min(ganancia)]

cat("\n=== MEJOR CONFIGURACIÓN ENCONTRADA ===\n")
if("iteracion_bayesiana" %in% names(tabla_log)) {
  cat("Iteración:", mejor_iteracion$iteracion_bayesiana, "de", nrow(tabla_log), "\n")
} else {
  cat("Iteración:", which.min(tabla_log$ganancia), "de", nrow(tabla_log), "\n")
}
cat("RMSE del mejor modelo:", mejor_iteracion$ganancia, "\n")

# Mostrar parámetros
cat("\nParámetros:\n")
for(param in columnas_requeridas[columnas_requeridas != "ganancia"]) {
  cat("  ", param, ":", mejor_iteracion[[param]], "\n")
}
if("num_iterations" %in% names(mejor_iteracion)) {
  cat("  num_iterations:", mejor_iteracion$num_iterations, "\n")
}

#------------------------------------------------------------------------------
# PARTE 2: Cargar datasets necesarios
#------------------------------------------------------------------------------

cat("\n=== CARGANDO DATOS ===\n")

# Primero, cargar el dataset original para evaluación posterior
setwd(carpeta_base)
setwd("./data/datasets")
dataset_original <- fread(PARAMS$feature_engineering$files$input$dentrada)
cat("Dataset original cargado:", nrow(dataset_original), "filas\n")
cat("Rango foto_mes en dataset original: [", min(dataset_original$foto_mes), ",", max(dataset_original$foto_mes), "]\n")

# Volver al directorio del experimento
setwd(carpeta_base)
setwd("./exp")
setwd(experiment_dir)
setwd(experiment_lead_dir)
setwd("03_HT")

#------------------------------------------------------------------------------
# PARTE 3: Entrenar modelo final (EXACTAMENTE como el HT)
#------------------------------------------------------------------------------

cat("\n=== ENTRENANDO MODELO FINAL CON MEJORES PARÁMETROS ===\n")

# Construir los parámetros del mejor modelo (como hace el HT)
mejores_params <- list()

# Primero los parámetros fijos del YAML
param_fijos <- PARAMS$hyperparameter_tuning$param$lightgbm
for(nombre in names(param_fijos)) {
  if(!nombre %in% c("num_iterations", "early_stopping_rounds")) {
    mejores_params[[nombre]] <- param_fijos[[nombre]]
  }
}

# Luego los parámetros optimizados
for(col in columnas_requeridas[columnas_requeridas != "ganancia"]) {
  if(col %in% names(mejor_iteracion)) {
    valor <- mejor_iteracion[[col]]
    # Convertir a entero si es necesario
    if(col %in% c("num_leaves", "min_data_in_leaf")) {
      mejores_params[[col]] <- as.integer(valor)
    } else {
      mejores_params[[col]] <- valor
    }
  }
}

# Agregar num_iterations si existe
if("num_iterations" %in% names(mejor_iteracion)) {
  mejores_params$num_iterations <- as.integer(mejor_iteracion$num_iterations)
} else {
  mejores_params$num_iterations <- 1000  # Default
}

# IMPORTANTE: No usar early stopping en modelo final (como hace el HT)
mejores_params$early_stopping_rounds <- NULL

cat("RMSE del mejor modelo:", mejor_iteracion$ganancia, "\n")

# Cargar dataset_final (como hace el HT)
dataset_final <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$train_final))
cat("\nDataset train_final cargado:", nrow(dataset_final), "filas\n")

max_mes_train <- max(dataset_final$foto_mes)
cat("Rango foto_mes: [", min(dataset_final$foto_mes), ",", max_mes_train, "]\n")

# Definir campos buenos del dataset_final (EXACTAMENTE como el HT)
campos_buenos_final <- setdiff(copy(colnames(dataset_final)),
                               c(PARAMS$hyperparameter_tuning$const$campo_clase))

# Crear dataset LightGBM
dtrain_final <- lgb.Dataset(data = data.matrix(dataset_final[, campos_buenos_final, with=FALSE]),
                           label = dataset_final[[PARAMS$hyperparameter_tuning$const$campo_clase]],
                           free_raw_data = FALSE)

# Entrenar modelo (como el HT)
set.seed(mejores_params$seed)
modelo_final <- lgb.train(data = dtrain_final,
                         param = mejores_params,
                         verbose = 100)

# Guardar el modelo final (como el HT)
saveRDS(modelo_final, file = "modelo_final_lgb.rds")
cat("\nModelo final guardado como: modelo_final_lgb.rds\n")

# Guardar la tabla de importancia final (como el HT)
tb_importancia_final <- as.data.table(lgb.importance(modelo_final))
fwrite(tb_importancia_final,
       file = PARAMS$hyperparameter_tuning$files$output$tb_importancia,
       sep = "\t")
cat("Importancia de variables guardada como:", PARAMS$hyperparameter_tuning$files$output$tb_importancia, "\n")

#------------------------------------------------------------------------------
# PARTE 4: Generar predicciones para datos presentes (EXACTAMENTE como el HT)
#------------------------------------------------------------------------------

cat("\n=== PREDICCIONES PARA DATOS PRESENTES ===\n")

present_file <- paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data)

if(file.exists(present_file)) {
  dataset_present <- fread(present_file)
  
  if(nrow(dataset_present) > 0) {
    cat("Dataset present cargado:", nrow(dataset_present), "filas\n")
    
    # Verificar meses en present
    meses_present <- unique(dataset_present$foto_mes)
    cat("Mes(es) en present:", paste(meses_present, collapse=", "), "\n")
    
    # Usar solo campos que existen en present (como el HT)
    campos_present <- intersect(campos_buenos_final, names(dataset_present))
    
    # Generar predicciones
    predicciones_present <- predict(modelo_final, 
                                   data.matrix(dataset_present[, campos_present, with=FALSE]))
    
    # Agregar predicciones al dataset (como el HT)
    dataset_present[, prediccion_clase := predicciones_present]
    
    # Guardar predicciones (como el HT)
    fwrite(dataset_present, file = "predicciones_presente.csv")
    cat("Predicciones para datos presentes guardadas como: predicciones_presente.csv\n")
    
    #--------------------------------------------------------------------------
    # PARTE ADICIONAL: Evaluación contra valores reales (no está en HT original)
    #--------------------------------------------------------------------------
    
    cat("\n=== EVALUACIÓN CONTRA VALORES REALES ===\n")
    
    # Tomar el mes más reciente de present
    if(length(meses_present) > 1) {
      mes_present <- max(meses_present)
      cat("Usando mes más reciente:", mes_present, "\n")
    } else {
      mes_present <- meses_present[1]
    }
    
    # Calcular el mes real a comparar
    mes_real <- sumar_meses(mes_present, orden_lead)
    cat("Mes con valores reales para comparar (present + lead):", mes_real, "\n")
    
    # Obtener valores reales del dataset original
    dataset_mes_real <- dataset_original[foto_mes == mes_real]
    
    if(nrow(dataset_mes_real) > 0) {
      cat("Datos reales encontrados para mes", mes_real, ":", nrow(dataset_mes_real), "filas\n")
      
      # Preparar tabla de evaluación
      dt_predicciones <- data.table(
        ticker = dataset_present$ticker,
        foto_mes_present = dataset_present$foto_mes,
        prediccion = dataset_present$prediccion_clase
      )
      
      dt_reales <- data.table(
        ticker = dataset_mes_real$ticker,
        foto_mes_real = dataset_mes_real$foto_mes,
        real = dataset_mes_real[[PARAMS$feature_engineering$const$origen_clase]]
      )
      
      # Merge por ticker
      dt_evaluacion <- merge(dt_predicciones[foto_mes_present == mes_present], 
                            dt_reales, 
                            by = "ticker", 
                            all.x = FALSE)
      
      if(nrow(dt_evaluacion) > 0) {
        cat("Tickers con predicción y valor real:", nrow(dt_evaluacion), "\n")
        
        # Calcular métricas
        dt_evaluacion[, error := prediccion - real]
        dt_evaluacion[, error_abs := abs(error)]
        dt_evaluacion[, error_pct := ifelse(real != 0, error / abs(real) * 100, NA)]
        
        rmse_test <- sqrt(mean(dt_evaluacion$error^2))
        mae_test <- mean(dt_evaluacion$error_abs)
        mape_test <- mean(abs(dt_evaluacion$error_pct), na.rm = TRUE)
        
        cat("\nMÉTRICAS DE EVALUACIÓN REAL:\n")
        cat("  Prediciendo desde mes:", mes_present, "\n")
        cat("  Evaluando en mes:", mes_real, "\n")
        cat("  Lead time:", orden_lead, "mes(es)\n")
        cat("  RMSE:", rmse_test, "\n")
        cat("  MAE:", mae_test, "\n")
        cat("  MAPE:", mape_test, "%\n")
        
        # Comparación con validación
        cat("\nComparación con validación:\n")
        cat("  RMSE en validación (BO):", mejor_iteracion$ganancia, "\n")
        cat("  RMSE en evaluación real:", rmse_test, "\n")
        cat("  Diferencia:", rmse_test - mejor_iteracion$ganancia, "\n")
        
        # Guardar evaluación
        fwrite(dt_evaluacion, file = "evaluacion_real.csv")
        cat("\nEvaluación real guardada como: evaluacion_real.csv\n")
        
        # Crear visualización
        if(nrow(dt_evaluacion) > 20) {
          cat("\n=== CREANDO VISUALIZACIÓN ===\n")
          
          p <- dt_evaluacion %>% 
            ggplot(aes(real, prediccion)) +
            geom_point(alpha = 0.6, size = 2) +
            geom_abline(slope = 1, linetype = "dashed", color = "red", alpha = 0.5) +
            scale_x_log10() +
            scale_y_log10() +
            theme_minimal() +
            labs(
              x = "Valor Real (log scale)",
              y = "Predicción (log scale)",
              title = paste("Predicción vs Real - Mes", mes_real),
              subtitle = paste("Modelo entrenado hasta:", max_mes_train, "| Present:", mes_present),
              caption = paste("RMSE:", round(rmse_test, 2), "| MAE:", round(mae_test, 2))
            )
          
          ggsave("prediccion_vs_real.png", p, width = 10, height = 8, dpi = 150)
          cat("Gráfico guardado como: prediccion_vs_real.png\n")
        }
        
        # Top errores
        cat("\nTop 5 mayores errores absolutos:\n")
        print(dt_evaluacion[order(-error_abs)][1:min(5, .N), 
              .(ticker, real, prediccion, error_abs, error_pct)])
        
        cat("\nTop 5 mejores predicciones:\n")
        print(dt_evaluacion[order(error_abs)][1:min(5, .N), 
              .(ticker, real, prediccion, error_abs, error_pct)])
        
      } else {
        cat("No hay tickers en común entre present y mes", mes_real, "\n")
      }
    } else {
      cat("No se encontraron datos para el mes", mes_real, "en dataset original\n")
      cat("Meses disponibles:", paste(sort(unique(dataset_original$foto_mes)), collapse=", "), "\n")
    }
    
  } else {
    cat("Dataset present está vacío\n")
  }
} else {
  cat("No se encontró archivo present_data:", present_file, "\n")
}

dt_evaluacion %>% 
  ggplot(aes(real, prediccion, fill = ticker, group = ticker)) +
  geom_point(shape = 21, size = 3, alpha = .6) +
  geom_label_repel(
    aes(label = ticker),
    size = 2.5,
    alpha = 0.9,
    label.padding = unit(0.15, "lines"),  # Padding dentro de la caja
    label.size = 0.15,                    # Grosor del borde de la caja
    box.padding = 0.3,
    point.padding = 0.2,
    segment.color = 'grey40',
    segment.size = 0.3,
    segment.alpha = 0.6,
    force = 2,                            # Fuerza de repulsión entre etiquetas
    force_pull = 1,                       # Fuerza de atracción al punto
    max.overlaps = Inf,                   # Mostrar todas las etiquetas
    seed = 42
  ) +
  scale_x_log10() +
  scale_y_log10() +
  geom_abline(slope = 1, linetype = "dashed", color = "red", alpha = 0.5) +
  theme(legend.position = "none") +
  labs(
    x = "Valor Real (log scale)",
    y = "Predicción (log scale)",
    title = paste("Predicción vs Real - Mes", mes_real),
    subtitle = paste("Modelo entrenado hasta:", max_mes_train, "| Present:", mes_present),
    caption = paste("RMSE:", round(rmse_test, 2), "| MAE:", round(mae_test, 2))
  )
#------------------------------------------------------------------------------
# PARTE 5: Guardar histórico y resumen final
#------------------------------------------------------------------------------

cat("\n=== GUARDANDO RESULTADOS ADICIONALES ===\n")

# Guardar histórico de optimización
fwrite(tabla_log, file = "historico_optimizacion.csv")
cat("Histórico de optimización guardado como: historico_optimizacion.csv\n")

# Crear resumen completo
resumen <- list(
  experimento = experiment_lead_dir,
  fecha_ejecucion = Sys.time(),
  optimizacion_bayesiana = list(
    iteraciones_totales = nrow(tabla_log),
    mejor_iteracion = which.min(tabla_log$ganancia),
    mejor_rmse = mejor_iteracion$ganancia,
    primera_ganancia = tabla_log$ganancia[1],
    mejora_total = tabla_log$ganancia[1] - mejor_iteracion$ganancia,
    mejora_porcentual = (tabla_log$ganancia[1] - mejor_iteracion$ganancia) / tabla_log$ganancia[1] * 100
  ),
  mejores_parametros = mejores_params,
  datos = list(
    n_train_final = nrow(dataset_final),
    max_mes_train = max_mes_train,
    n_present = ifelse(exists("dataset_present"), nrow(dataset_present), 0),
    mes_present = ifelse(exists("mes_present"), mes_present, NA)
  ),
  evaluacion = if(exists("rmse_test")) {
    list(
      mes_evaluado = mes_real,
      rmse = rmse_test,
      mae = mae_test,
      mape = mape_test,
      n_evaluados = nrow(dt_evaluacion),
      diferencia_vs_validacion = rmse_test - mejor_iteracion$ganancia
    )
  } else {
    NULL
  }
)

saveRDS(resumen, file = "resumen_evaluacion.rds")
cat("Resumen guardado como: resumen_evaluacion.rds\n")

# Mostrar importancia de variables
cat("\n=== TOP 10 VARIABLES MÁS IMPORTANTES ===\n")
print(head(tb_importancia_final[order(-Gain)], 10))

cat("\n=== PROCESO COMPLETADO ===\n")
cat("Directorio de trabajo:", getwd(), "\n")
cat("\nArchivos generados:\n")
cat("  - modelo_final_lgb.rds (modelo final)\n")
cat("  - ", PARAMS$hyperparameter_tuning$files$output$tb_importancia, "(importancia de variables)\n")
if(file.exists("predicciones_presente.csv")) {
  cat("  - predicciones_presente.csv (predicciones para present)\n")
}
if(exists("dt_evaluacion") && nrow(dt_evaluacion) > 0) {
  cat("  - evaluacion_real.csv (evaluación contra valores reales)\n")
  cat("  - prediccion_vs_real.png (gráfico de evaluación)\n")
}
cat("  - historico_optimizacion.csv (histórico de BO)\n")
cat("  - resumen_evaluacion.rds (resumen completo)\n")

rm(tabla_log)
cat("\n¡Proceso de evaluación desde BO_log.txt completado exitosamente!\n")
