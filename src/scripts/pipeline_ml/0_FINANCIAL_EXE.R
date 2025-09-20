#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

library(data.table)
library(lubridate)

#################### Definicion Parametros ######################
library(yaml)
carpeta_base <- "~/labo-ml-befar"
setwd(carpeta_base)
objetos_trans_script <- c("experiment_dir","experiment_lead_dir","carpeta_base","objetos_trans_script")

PARAMS <- yaml.load_file("./src/scripts/pipeline_ml/0_FINANCIAL_YML.yml")

# Carpetas de experimento
experiment_dir <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,sep = "_")
experiment_lead_dir <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,paste0("f",PARAMS$feature_engineering$const$orden_lead),sep = "_")

# Crear directorios en bucket
bucket_exp <- "~/buckets/b1/exp"
dir.create(bucket_exp, showWarnings = FALSE, recursive = TRUE)
setwd(bucket_exp)

dir.create(experiment_dir,showWarnings = FALSE)
setwd(experiment_dir)
dir.create(experiment_lead_dir,showWarnings = FALSE)
setwd(experiment_lead_dir)

#################### Redefinir tiempos para financial data ##########################
# Para datos financieros, trabajamos con meses en formato numérico YYYYMM
presente_mes <- PARAMS$feature_engineering$const$presente
canaritos_mes_end <- 202408  # Un mes antes del lead
PARAMS$feature_engineering$const$canaritos_mes_end <- canaritos_mes_end

canaritos_mes_valid <- 202409  # Mes de validación
PARAMS$feature_engineering$const$canaritos_mes_valid <- canaritos_mes_valid
#-------------------------------------------------------------------------

#################################################################
# Persisto los parametros en un json
jsontest = jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE)
write(jsontest,paste0(experiment_lead_dir,".json"))

################### Feature Engineering ################### 
setwd(carpeta_base)
setwd("./src/scripts/pipeline_ml")
source(PARAMS$feature_engineering$files$fe_script)

################### Training Strategy ###################

#limpio la memoria
rm( list=setdiff(ls(),objetos_trans_script) )  #remove objects
gc()             #garbage collection

setwd(bucket_exp)
setwd(experiment_dir)
setwd(experiment_lead_dir)

jsonfile <- list.files(pattern = ".json")
PARAMS <- jsonlite::fromJSON(jsonfile)

setwd(carpeta_base)
setwd("./src/scripts/pipeline_ml")
source(PARAMS$training_strategy$files$ts_script)

#################################################################
setwd(bucket_exp)
setwd(experiment_dir)
setwd(experiment_lead_dir)
# actualizo los parametros del json
jsontest = jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE)
write(jsontest,paste0(experiment_lead_dir,".json"))

################### Hyperparameter Tuning ###################
#limpio la memoria
rm( list=setdiff(ls(),objetos_trans_script) )  #remove objects
gc()             #garbage collection

setwd(bucket_exp)
setwd(experiment_dir)
setwd(experiment_lead_dir)

jsonfile <- list.files(pattern = ".json")
PARAMS <- jsonlite::fromJSON(jsonfile)

setwd(carpeta_base)
setwd("./src/scripts/pipeline_ml")

source(PARAMS$hyperparameter_tuning$files$ht_script)

cat("Pipeline Financial Forecasting completado exitosamente!\n")
