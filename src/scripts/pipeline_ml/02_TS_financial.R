# Training Strategy para Financial Forecasting - EXACTA RÉPLICA de 02_TS_generico.R
# Adaptado para usar parámetros de 0_FINANCIAL_YML.yml
# Este script replica EXACTAMENTE la lógica de 02_TS_generico.R para financial forecasting

#Arma la Training Strategy
#parte del dataset con feature engineering
#  1. dataset de present, donde se va a aplicar el modelo, son los datos que NO tiene clase
#  2. dataset de la train_strategy, donde estan marcados los registros de training, validation y testing
#        puede no considerar algunos años, hacer undersampling de las clases, etc
#        hace falta mucha creatividad para esta estapa, decidir donde entrar es MUY ESTRATEGICO y esta sometido a las limitaciones de procesamiento
#  3. dataset de  train_final  donde se va a entrenar el modelo final una vez que se tengan los mejores hiperparametros de la Bayesian Optimization

require("data.table")
require("yaml")
library(lubridate)

#------------------------------------------------------------------------------
#particiona en el dataset una seccion  del yaml

aplicar_particion  <- function( seccion )
{
  columna_nueva  <- paste0( "part_", seccion)
  dataset[  , (columna_nueva) := 0L ]

  if( length( PARAMS$training_strategy$param[[seccion]]$periodos ) > 0 )
  {
    dataset[ get( PARAMS$training_strategy$const$periodo ) %in%  PARAMS$training_strategy$param[[seccion]]$periodos ,
             (columna_nueva)  := 1L ]
  } else {

     dataset[ get( PARAMS$training_strategy$const$periodo ) >= PARAMS$training_strategy$param[[seccion]]$rango$desde  &  get(PARAMS$training_strategy$const$periodo)  <= PARAMS$training_strategy$param[[seccion]]$rango$hasta,
              (columna_nueva)  := 1L ]

  }

  if( length( PARAMS$training_strategy$param[[seccion]]$excluir ) > 0 )
  {
    dataset[ get( PARAMS$training_strategy$const$periodo ) %in%  PARAMS$training_strategy$param[[seccion]]$excluir , 
             (columna_nueva) := 0L ]
  }


  if( "undersampling" %in% names( PARAMS$training_strategy$param[[seccion]] ) )
  {
    for( clase_valor  in  PARAMS$training_strategy$param[[seccion]]$undersampling )
    {

       dataset[ get(columna_nueva)==1L & get(PARAMS$training_strategy$const$clase) == clase_valor$clase  & part_azar > clase_valor$prob,
                (columna_nueva) := 0L  ]
                  
    }
  }

}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa
#el input es SIEMPRE un dataset con Feature Engineering

# Cargar parámetros YAML
PARAMS <- yaml.load_file("0_FINANCIAL_YML.yml")

# Establecer directorio base
setwd(PARAMS$environment$base_dir)

#cargo el dataset
nom_exp_folder <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,sep = "_")

nom_subexp_folder <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,paste0("f",PARAMS$feature_engineering$const$orden_lead),sep = "_")

nom_arch  <- paste0(paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,paste0("f",PARAMS$feature_engineering$const$orden_lead),sep = "_"),".csv.gz" )

# Navegar al directorio del dataset
setwd(paste0("./exp/",nom_exp_folder,"/",nom_subexp_folder,"/01_FE"))
dataset   <- fread( nom_arch )

# Eliminar todas las variables que contengan la cadena "hf3"
dataset <- dataset[, !grepl("hf3", names(dataset)), with = FALSE]
################# Redefinir periodos ##########################
# Función helper para convertir foto_mes a fecha y hacer operaciones
mes_to_date <- function(foto_mes) {
  año <- foto_mes %/% 100
  mes <- foto_mes %% 100
  return(as.Date(paste0(año, "-", sprintf("%02d", mes), "-01")))
}

date_to_mes <- function(fecha) {
  año <- year(fecha)
  mes <- month(fecha)
  return(año * 100 + mes)
}

# Calcular el mes límite (presente - orden_lead meses)
presente_date <- mes_to_date(PARAMS$feature_engineering$const$presente)
limite_date <- presente_date %m-% months(PARAMS$feature_engineering$const$orden_lead)
limite_mes <- date_to_mes(limite_date)

# Test: máximo foto_mes disponible hasta el límite
PARAMS$training_strategy$param$test$periodos <- max(unique(setdiff(
  dataset[foto_mes <= limite_mes]$foto_mes,
  PARAMS$training_strategy$param$train$excluir
)))

# Validate: 1 mes antes del test
test_date <- mes_to_date(PARAMS$training_strategy$param$test$periodos)
validate_date <- test_date %m-% months(1)
PARAMS$training_strategy$param$validate$periodos <- date_to_mes(validate_date)

# Train hasta: 2 meses antes del test
train_date <- test_date %m-% months(2)
PARAMS$training_strategy$param$train$rango$hasta <- date_to_mes(train_date)

# Train final hasta: igual al test
PARAMS$training_strategy$param$train_final$rango$hasta <- PARAMS$training_strategy$param$test$periodos
###############################################################

#ordeno el dataset por <foto_mes, ticker> - ADAPTADO para financial forecasting
setorderv( dataset, PARAMS$training_strategy$const$campos_sort )

set.seed( PARAMS$training_strategy$param$semilla )  #uso la semilla
dataset[ , part_azar  := runif( nrow(dataset) ) ]   #genero variable azar, con la que voy a particionar

#hago las particiones de cada seccion
#las secciones son  present,   train, validate, test,  train_final

for( seccion  in  PARAMS$training_strategy$const$secciones ){
  aplicar_particion( seccion )
}

dataset[ , part_azar := NULL ]  #ya no necesito el campo  part_azar

psecciones  <- paste0( "part_", PARAMS$training_strategy$const$secciones )

#genero el archivo de control, que DEBE ser revisado
tb_control  <- dataset[ , .N, 
                        psecciones]

# Mostrar resumen de particiones
cat("\n=== RESUMEN DE PARTICIONES ===\n")
print(tb_control)

# Verificar que hay registros en cada partición crítica
cat("\n=== VERIFICACIÓN DE PARTICIONES ===\n")
cat("Present (sin clase):", dataset[ part_present>0, .N ], "registros\n")
cat("Train:", dataset[ part_train>0, .N ], "registros\n") 
cat("Validate:", dataset[ part_validate>0, .N ], "registros\n")
cat("Test:", dataset[ part_test>0, .N ], "registros\n")
cat("Train_final:", dataset[ part_train_final>0, .N ], "registros\n")

# Crear directorio de salida
setwd("../")
dir.create("02_TS",showWarnings = FALSE)
setwd("02_TS")

# Guardar archivo de control
fwrite( tb_control,
        file= paste0(PARAMS$training_strategy$files$output$control),
        sep= "\t" )

cat("\nArchivo de control guardado:", PARAMS$training_strategy$files$output$control, "\n")

#Grabo present (datos sin clase para predicción)
if( 0 < dataset[ part_present>0, .N ] ){
  #Grabo present
  fwrite( dataset[ part_present>0,
                   setdiff( colnames( dataset ) , 
                            c( psecciones, PARAMS$training_strategy$const$clase ) ),
                   with= FALSE ] ,
        file= paste0(PARAMS$training_strategy$files$output$present_data),
        logical01 = TRUE,
        sep= "," )
  
  cat("Present data guardado:", PARAMS$training_strategy$files$output$present_data, 
      "con", dataset[ part_present>0, .N ], "registros\n")
}

#Grabo train_strategy (train + validate + test)
if( 0 < dataset[ part_train>0 | part_validate>0 | part_test>0, .N ] ){
  fwrite( dataset[ part_train>0 | part_validate>0 | part_test>0,
                   setdiff( colnames( dataset ) , c("part_present","part_train_final") ),
                   with= FALSE ] ,
          file= paste0(PARAMS$training_strategy$files$output$train_strategy),
          logical01 = TRUE,
          sep= "," )
  
  cat("Train strategy guardado:", PARAMS$training_strategy$files$output$train_strategy,
      "con", dataset[ part_train>0 | part_validate>0 | part_test>0, .N ], "registros\n")
}

#Grabo train_final (para entrenamiento final del modelo)
if( 0 < dataset[ part_train_final > 0, .N ] ){
  fwrite( dataset[ part_train_final > 0,
                   setdiff( colnames( dataset ),
                            psecciones ),
                   with= FALSE ] ,
          file= paste0(PARAMS$training_strategy$files$output$train_final),
          logical01 = TRUE,
          sep= "," )
  
  cat("Train final guardado:", PARAMS$training_strategy$files$output$train_final,
      "con", dataset[ part_train_final > 0, .N ], "registros\n")
}

cat("\n=== TRAINING STRATEGY FINANCIAL FORECASTING COMPLETADO ===\n")
cat("Archivos generados en directorio: 02_TS\n")
cat("Revise el archivo de control para verificar las particiones\n")
