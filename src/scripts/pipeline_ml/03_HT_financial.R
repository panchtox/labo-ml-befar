# 03_HT_financial.R - Optimización Bayesiana LightGBM para Financial Forecasting
# Adaptado de 03_HT_generico.R para funcionar con pipeline financial

require("data.table")
require("primes")
require("lightgbm")

# Paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")
require("rlist")

setDTthreads(percent = 65)

#------------------------------------------------------------------------------
# Graba a un archivo los componentes de lista
# Para el primer registro, escribe antes los títulos

loguear <- function(reg, arch=NA, folder="./exp/", ext=".txt", verbose=TRUE) {
  archivo <- arch
  if(is.na(arch)) archivo <- paste0(folder, substitute(reg), ext)
  
  if(!file.exists(archivo)) { # Escribo los títulos
    linea <- paste0("fecha\t", 
                    paste(list.names(reg), collapse="\t"), "\n")
    cat(linea, file=archivo)
  }
  
  linea <- paste0(format(Sys.time(), "%Y%m%d %H%M%S"), "\t", # la fecha y hora
                  gsub(", ", "\t", toString(reg)), "\n")
  
  cat(linea, file=archivo, append=TRUE) # grabo al archivo
  
  if(verbose) cat(linea) # imprimo por pantalla
}

#------------------------------------------------------------------------------

parametrizar <- function(lparam) {
  param_fijos <- copy(lparam)
  hs <- list()

  for(param in names(lparam)) {
    if(length(lparam[[param]]) > 1) {
      desde <- as.numeric(lparam[[param]][[1]])
      hasta <- as.numeric(lparam[[param]][[2]])

      if(length(lparam[[param]]) == 2) {
        hs <- append(hs, 
                     list(makeNumericParam(param, lower=desde, upper=hasta)))
      } else {
        hs <- append(hs, 
                     list(makeIntegerParam(param, lower=desde, upper=hasta)))
      }

      param_fijos[[param]] <- NULL # lo quito 
    }
  }

  return(list("param_fijos" = param_fijos,
              "paramSet" = hs))
}

#------------------------------------------------------------------------------
# Particiona un dataset en forma estratificada

particionar <- function(data, division, agrupa="", campo="fold", start=1, seed=NA) {
  if(!is.na(seed)) set.seed(seed)

  bloque <- unlist(mapply(function(x,y) { rep(y, x) }, division, seq(from=start, length.out=length(division))))  

  data[, (campo) := sample(rep(bloque, ceiling(.N/length(bloque))))[1:.N],
       by=agrupa]
}

#------------------------------------------------------------------------------

EstimarGanancia_lightgbm <- function(x) {
  gc()
  GLOBAL_iteracion <<- GLOBAL_iteracion + 1

  param_completo <- c(param_fijos, x)

  param_completo$num_iterations <- ifelse(param_fijos$boosting == "dart", 999, 99999) # un numero muy grande
  param_completo$early_stopping_rounds <- as.integer(200 + 4/param_completo$learning_rate)

  set.seed(param_completo$seed)
  modelo_train <- lgb.train(data = dtrain,
                           valids = list(valid = dvalidate),
                           param = param_completo,
                           verbose = -100)

  # Aplico el modelo a testing y calculo la métrica
  prediccion <- predict(modelo_train, 
                       data.matrix(dataset_test[, campos_buenos, with=FALSE]))

  tbl <- dataset_test[, c(PARAMS$hyperparameter_tuning$const$campo_clase), with=F]
  tbl[, pred := prediccion]

  gc()

  # Para financial forecasting: usar la métrica configurada (RMSE por defecto)
  parametro <- unlist(modelo_train$record_evals$valid[[PARAMS$hyperparameter_tuning$param$lightgbm$metric]]$eval)[modelo_train$best_iter]

  ganancia_test_normalizada <- parametro

  rm(tbl)
  gc()
  
  # Voy grabando las mejores column importance
  if(ganancia_test_normalizada < GLOBAL_ganancia) {
    GLOBAL_ganancia <<- ganancia_test_normalizada
    tb_importancia <- as.data.table(lgb.importance(modelo_train))

    fwrite(tb_importancia,
           file = paste0(PARAMS$hyperparameter_tuning$files$output$importancia, GLOBAL_iteracion, ".txt"),
           sep = "\t")
  }

  # Logueo final
  ds <- list("cols" = ncol(dtrain), "rows" = nrow(dtrain))
  xx <- c(ds, copy(param_completo))

  xx$early_stopping_rounds <- NULL
  xx$num_iterations <- modelo_train$best_iter
  xx$ganancia <- ganancia_test_normalizada
  xx$iteracion_bayesiana <- GLOBAL_iteracion

  loguear(xx, arch = PARAMS$hyperparameter_tuning$files$output$BOlog)

  return(ganancia_test_normalizada)
}

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aquí empieza el programa

set.seed(PARAMS$hyperparameter_tuning$param$semilla) # dejo fija esta semilla

setwd(paste0(carpeta_base, "/exp"))
setwd(experiment_dir)
setwd(experiment_lead_dir)
setwd("02_TS")

# Cargo el dataset que tiene la Training Strategy
nom_arch <- PARAMS$hyperparameter_tuning$files$input$dentrada
dataset <- fread(nom_arch)

# Creo la carpeta donde va el experimento
# HT representa Hiperparameter Tuning
setwd(paste0(carpeta_base, "/exp"))
setwd(experiment_dir)
setwd(experiment_lead_dir)
dir.create("03_HT", showWarnings = FALSE)
setwd("03_HT")

# Los campos que se pueden utilizar para la predicción
campos_buenos <- setdiff(copy(colnames(dataset)),
                        c(PARAMS$hyperparameter_tuning$const$campo_clase, 
                          "part_train", "part_validate", "part_test"))

# La partición de train siempre va
dtrain <- lgb.Dataset(data = data.matrix(dataset[part_train==1, campos_buenos, with=FALSE]),
                     label = dataset[part_train==1][[PARAMS$hyperparameter_tuning$const$campo_clase]],
                     free_raw_data = FALSE)

# Calculo validation y testing, según corresponda
if(PARAMS$hyperparameter_tuning$param$crossvalidation == FALSE) {
  if(PARAMS$hyperparameter_tuning$param$validate == TRUE) {
    dvalidate <- lgb.Dataset(data = data.matrix(dataset[part_validate==1, campos_buenos, with=FALSE]),
                            label = dataset[part_validate==1][[PARAMS$hyperparameter_tuning$const$campo_clase]],
                            free_raw_data = FALSE)

    dataset_test <- dataset[part_test == 1]
    test_multiplicador <- 1

  } else {
    # Divido en mitades los datos de testing
    particionar(dataset, 
                division = c(1,1),
                agrupa = c("part_test", PARAMS$hyperparameter_tuning$const$campo_periodo), 
                seed = PARAMS$hyperparameter_tuning$param$semilla,
                campo = "fold_test")

    # fold_test==1 lo tomo para validation
    dvalidate <- lgb.Dataset(data = data.matrix(dataset[part_test==1 & fold_test==1, campos_buenos, with=FALSE]),
                            label = dataset[part_test==1 & fold_test==1, PARAMS$hyperparameter_tuning$const$campo_clase],
                            free_raw_data = FALSE)

    dataset_test <- dataset[part_test==1 & fold_test==2, ]
    test_multiplicador <- 2
  }
}

rm(dataset)
gc()

# Prepara todo la la Bayesian Optimization -------------------------------------
hiperparametros <- PARAMS$hyperparameter_tuning$param[[PARAMS$hyperparameter_tuning$param$algoritmo]]
apertura <- parametrizar(hiperparametros)
param_fijos <- apertura$param_fijos

# Si ya existe el archivo log, traigo hasta donde procesé
if(file.exists(PARAMS$hyperparameter_tuning$files$output$BOlog)) {
  tabla_log <- fread(PARAMS$hyperparameter_tuning$files$output$BOlog)
  GLOBAL_iteracion <- nrow(tabla_log)
  GLOBAL_ganancia <- tabla_log[, min(ganancia)]
  rm(tabla_log)
} else {
  GLOBAL_iteracion <- 0
  GLOBAL_ganancia <- Inf
}

# Aquí comienza la configuración de mlrMBO
funcion_optimizar <- EstimarGanancia_lightgbm

configureMlr(show.learner.output = FALSE)

# Configuro la búsqueda bayesiana, los hiperparámetros que se van a optimizar
obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar, # la función que voy a optimizar
  minimize = PARAMS$hyperparameter_tuning$param$BO$minimize, # minimizar RMSE
  noisy = PARAMS$hyperparameter_tuning$param$BO$noisy,
  par.set = makeParamSet(params = apertura$paramSet), # definido al comienzo del programa
  has.simple.signature = PARAMS$hyperparameter_tuning$param$BO$has.simple.signature # paso los parámetros en una lista
)

# Archivo donde se graba y cada cuántos segundos
ctrl <- makeMBOControl(save.on.disk.at.time = PARAMS$hyperparameter_tuning$param$BO$save.on.disk.at.time,  
                      save.file.path = PARAMS$hyperparameter_tuning$files$output$BObin)
                         
ctrl <- setMBOControlTermination(ctrl, 
                                iters = PARAMS$hyperparameter_tuning$param$BO$iterations) # cantidad de iteraciones
                                   
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

# Establezco la función que busca el óptimo
surr.km <- makeLearner("regr.km",
                      predict.type = "se",
                      covtype = "matern3_2",
                      control = list(trace = TRUE))

# Aquí inicio la optimización bayesiana
if(!file.exists(PARAMS$hyperparameter_tuning$files$output$BObin)) {
  run <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  # Si ya existe el archivo RDATA, debo continuar desde el punto hasta donde llegué
  # Usado para cuando se corta la virtual machine
  run <- mboContinue(PARAMS$hyperparameter_tuning$files$output$BObin) # retomo en caso que ya exista
}

#------------------------------------------------------------------------------
# Entreno el modelo final con los mejores parámetros encontrados
#------------------------------------------------------------------------------

# Cargo los mejores parámetros del log
if(file.exists(PARAMS$hyperparameter_tuning$files$output$BOlog)) {
  tabla_log <- fread(PARAMS$hyperparameter_tuning$files$output$BOlog)
  mejor_iteracion <- tabla_log[which.min(ganancia)]
  
  # Construyo los parámetros del mejor modelo
  mejores_params <- param_fijos
  for(col in names(apertura$paramSet)) {
    if(col %in% names(mejor_iteracion)) {
      mejores_params[[col]] <- mejor_iteracion[[col]]
    }
  }
  
  mejores_params$num_iterations <- mejor_iteracion$num_iterations
  mejores_params$early_stopping_rounds <- NULL # No usar early stopping en modelo final
  
  cat("Entrenando modelo final con mejores parámetros...\n")
  cat("RMSE del mejor modelo:", mejor_iteracion$ganancia, "\n")
  
  # Entreno modelo final con train + validate
  dataset_final <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$train_final))
  
  campos_buenos_final <- setdiff(copy(colnames(dataset_final)),
                                c(PARAMS$hyperparameter_tuning$const$campo_clase))
  
  dtrain_final <- lgb.Dataset(data = data.matrix(dataset_final[, campos_buenos_final, with=FALSE]),
                             label = dataset_final[[PARAMS$hyperparameter_tuning$const$campo_clase]],
                             free_raw_data = FALSE)
  
  set.seed(mejores_params$seed)
  modelo_final <- lgb.train(data = dtrain_final,
                           param = mejores_params,
                           verbose = 100)
  
  # Guardo el modelo final
  saveRDS(modelo_final, file = "modelo_final_lgb.rds")
  
  # Guardo la tabla de importancia final
  tb_importancia_final <- as.data.table(lgb.importance(modelo_final))
  fwrite(tb_importancia_final,
         file = PARAMS$hyperparameter_tuning$files$output$tb_importancia,
         sep = "\t")
  
  cat("Modelo final guardado como: modelo_final_lgb.rds\n")
  cat("Importancia de variables guardada como:", PARAMS$hyperparameter_tuning$files$output$tb_importancia, "\n")
  
  # Genero predicciones para datos presentes (si existen)
  if(file.exists(paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data))) {
    dataset_present <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data))
    
    if(nrow(dataset_present) > 0) {
      campos_present <- intersect(campos_buenos_final, names(dataset_present))
      
      predicciones_present <- predict(modelo_final, 
                                     data.matrix(dataset_present[, campos_present, with=FALSE]))
      
      dataset_present[, prediccion_clase := predicciones_present]
      
      fwrite(dataset_present, file = "predicciones_presente.csv")
      cat("Predicciones para datos presentes guardadas como: predicciones_presente.csv\n")
    }
  }
  
  rm(tabla_log)
}

cat("Proceso de Hyperparameter Tuning completado.\n")
