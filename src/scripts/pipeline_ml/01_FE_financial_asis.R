# Feature Engineering para Financial Forecasting - EXACTA RÉPLICA de 01_FE_health_asis.R
# Adaptado para usar parámetros de 0_FINANCIAL_YML.yml
# El feature engineering es propio de cada dataset
# realiza Feature Engineering sobre el dataset original
# Este script replica EXACTAMENTE la lógica de 01_FE_health_asis.R

require("data.table")
require("Rcpp")
require("rlist")
require("yaml")
library(dplyr)
library(stringr)
library(lubridate)

require("lightgbm")
require("ranger")
require("randomForest")  #solo se usa para imputar nulos

#------------------------------------------------------------------------------

ReportarCampos  <- function( dataset )
{
  cat( "La cantidad de campos es ", ncol(dataset) , "\n" )
}
#------------------------------------------------------------------------------
#Agrega al dataset una variable que va de 1 a 4, el trimestre, para que el modelo aprenda estacionalidad
# ADAPTADO: Para financial data, creamos un indicador cíclico del trimestre desde foto_mes

AgregarTrimestre  <- function( dataset )
{
  gc()
  # Para financial data: crear variable cíclica GLOBAL por trimestre
  # Extraer el trimestre del formato numérico YYYYMM (ej: 202403 = 2024 marzo = Q1)
  dataset[, quarter_cycle := ceiling((foto_mes %% 100)/3)]  # Convierte mes a trimestre (1-4)
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Agrega al dataset una variable que va de 1 a 12, el mes, para que el modelo aprenda estacionalidad mensual
# NUEVO: Para financial data, creamos un indicador cíclico del mes

AgregarMes  <- function( dataset )
{
  gc()
  # Para financial data: crear variable cíclica GLOBAL por mes
  # Extraer el mes del formato numérico YYYYMM (ej: 202403 = marzo = 3)
  dataset[, month_cycle := foto_mes %% 100]  # Extrae mes (01-12)
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Elimina las variables que uno supone hace Data Drifting

DriftEliminar  <- function( dataset, variables )
{
  gc()
  dataset[  , c(variables) := NULL ]
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Autor:  Santiago Dellachiesa, UAustral 2021
#A las variables que tienen nulos, les agrega una nueva variable el dummy de is es nulo o no {0, 1}

DummiesNA  <- function( dataset )
{
  gc()
  # ADAPTADO: usar 'presente' como mes actual en lugar de foto_mes
  nulos  <- colSums( is.na(dataset[ foto_mes %in% PARAMS$feature_engineering$const$presente ]) )  #cuento la cantidad de nulos por columna
  colsconNA  <- names( which(  nulos > 0 ) )
  
  cat("Creando dummies para", length(colsconNA), "columnas con NAs:\n")
  if(length(colsconNA) > 0) {
    cat(paste(colsconNA[1:min(5, length(colsconNA))], collapse=", "))
    if(length(colsconNA) > 5) cat(" ...")
    cat("\n")
    
    # Crear dummies individuales para cada columna
    for(col in colsconNA) {
      dummy_name <- paste0(col, "_isNA")
      dataset[, (dummy_name) := as.numeric(is.na(get(col)))]
    }
  }

  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------

CorregirCampoMes  <- function( pcampo, pmeses )
{
  # ADAPTADO: usar ticker en lugar de Country Code
  tbl <- dataset[  ,  list( "v1" = shift( get(pcampo), 1, type="lag" ),
                            "v2" = shift( get(pcampo), 1, type="lead" )
                         ), 
                   by=ticker ]
  
  tbl[ , ticker := NULL ]
  tbl[ , promedio := rowMeans( tbl,  na.rm=TRUE ) ]
  
  dataset[ ,  
           paste0(pcampo) := ifelse( !(foto_mes %in% pmeses),
                                     get( pcampo),
                                     tbl$promedio ) ]

}
#------------------------------------------------------------------------------
# reemplaza cada variable ROTA  (variable, foto_q)  con el promedio entre  ( trimestre_anterior, trimestre_posterior )
# en honor a Claudio Castillo,  honorable y fiel centinela de la Estadistica Clasica

CorregirClaudioCastillo  <- function( dataset )
{
  # Para financial data, no hay correcciones específicas conocidas
  # Se mantiene la estructura pero sin correcciones activas
}
#------------------------------------------------------------------------------
#Corrige poniendo a NA las variables que en ese trimestre estan dañadas

CorregirNA  <- function( dataset )
{
  gc()
  #acomodo los errores del dataset - adaptado para financial data
  # Ejemplo: ciertos meses con datos problemáticos conocidos
  # dataset[ foto_mes==202003,  is_totalRevenue   := NA ]  # COVID impact

  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#Esta es la parte que los alumnos deben desplegar todo su ingenio
# ADAPTADO para financial forecasting

AgregarVariables  <- function( dataset )
{
  gc()
  
  # Diagnóstico de tipos de columnas
  cat("=== DIAGNÓSTICO DE COLUMNAS ===\n")
  tipos_cols <- sapply(dataset, class)
  cat("Tipos de columnas encontrados:\n")
  for(i in seq_along(tipos_cols)) {
    if(is.list(tipos_cols[[i]])) {
      cat(names(tipos_cols)[i], ":", paste(tipos_cols[[i]], collapse=", "), "\n")
    } else {
      cat(names(tipos_cols)[i], ":", tipos_cols[[i]], "\n")
    }
  }
  
  # Identificar columnas problemáticas
  cols_problema <- names(dataset)[sapply(dataset, function(x) is.list(x) || (!is.numeric(x) && !is.character(x) && !is.logical(x)))]
  if(length(cols_problema) > 0) {
    cat("COLUMNAS PROBLEMÁTICAS encontradas:", paste(cols_problema, collapse=", "), "\n")
  }
  cat("===============================\n\n")
  
  #INICIO de la seccion donde se deben hacer cambios con variables nuevas
  
  # 0. VARIABLES DE RETRASO COMO FEATURES (NUEVO PARA DATASET CONSOLIDADO)
  if("retraso_meses" %in% names(dataset)) {
    # Crear variables derivadas del retraso
    dataset[, retraso_trimestre := ceiling(retraso_meses / 3)]
    dataset[, es_dato_actual := as.numeric(retraso_meses == 0)]
    dataset[, es_dato_reciente := as.numeric(retraso_meses <= 3)]
  }
  
  # 1. Variables específicas de financial analysis usando parámetros YAML
  if(PARAMS$feature_engineering$param$financial_ratios) {
    # Ratio Deuda/Activos
    if("bs_totalLiabilities" %in% names(dataset) & "bs_totalAssets" %in% names(dataset)) {
      dataset[, debt_to_assets := bs_totalLiabilities / (bs_totalAssets + 1)]
    }
    
    # Ratio Activos/Patrimonio (Leverage)
    if("bs_totalAssets" %in% names(dataset) & "bs_totalShareholderEquity" %in% names(dataset)) {
      dataset[, asset_to_equity := bs_totalAssets / (bs_totalShareholderEquity + 1)]
    }
    
    # ROA (Return on Assets)
    if("is_netIncome" %in% names(dataset) & "bs_totalAssets" %in% names(dataset)) {
      dataset[, roa := is_netIncome / (bs_totalAssets + 1)]
    }
    
    # ROE (Return on Equity) 
    if("is_netIncome" %in% names(dataset) & "bs_totalShareholderEquity" %in% names(dataset)) {
      dataset[, roe := is_netIncome / (bs_totalShareholderEquity + 1)]
    }
  }
  
  if(PARAMS$feature_engineering$param$valuation_metrics) {
    # Price to Book ratio
    if("Close" %in% names(dataset) & "bs_totalShareholderEquity" %in% names(dataset) & "bs_commonStockSharesOutstanding" %in% names(dataset)) {
      dataset[, book_value_per_share := bs_totalShareholderEquity / (bs_commonStockSharesOutstanding + 1)]
      dataset[, price_to_book := Close / (book_value_per_share + 1)]
    }
    
    # Earnings per Share
    if("is_netIncome" %in% names(dataset) & "bs_commonStockSharesOutstanding" %in% names(dataset)) {
      dataset[, eps := is_netIncome / (bs_commonStockSharesOutstanding + 1)]
    }
    
    # P/E ratio
    if("Close" %in% names(dataset) & "eps" %in% names(dataset)) {
      dataset[, pe_ratio := Close / (eps + 0.01)]  # Evitar división por 0
    }
  }
  
  if(PARAMS$feature_engineering$param$profitability_ratios) {
    # Gross Margin
    if("is_grossProfit" %in% names(dataset) & "is_totalRevenue" %in% names(dataset)) {
      dataset[, gross_margin := is_grossProfit / (is_totalRevenue + 1)]
    }
    
    # Operating Margin
    if("is_operatingIncome" %in% names(dataset) & "is_totalRevenue" %in% names(dataset)) {
      dataset[, operating_margin := is_operatingIncome / (is_totalRevenue + 1)]
    }
    
    # Net Margin
    if("is_netIncome" %in% names(dataset) & "is_totalRevenue" %in% names(dataset)) {
      dataset[, net_margin := is_netIncome / (is_totalRevenue + 1)]
    }
  }
  
  # 2. Calcular el primer mes donde Close > 0 para cada ticker
  dataset[Close > 0, FirstMonth := min(foto_mes, na.rm = TRUE), by = ticker]
  
  # 3. Llenar los valores de FirstMonth para todas las filas
  dataset[, FirstMonth := nafill(FirstMonth, type = "locf"), by = ticker]
  dataset[, FirstMonth := nafill(FirstMonth, type = "nocb"), by = ticker]
  
  # 4. Calcular la diferencia en meses desde el primer mes
  # Función auxiliar para convertir YYYYMM a número de meses desde una base
  convertir_a_meses <- function(foto_mes_val, base_mes = 200501) {
    año_actual <- as.integer(foto_mes_val / 100)
    mes_actual <- foto_mes_val %% 100
    
    año_base <- as.integer(base_mes / 100)
    mes_base <- base_mes %% 100
    
    return((año_actual - año_base) * 12 + (mes_actual - mes_base))
  }
  
  dataset[, MonthsSinceFirst := ifelse(!is.na(FirstMonth), 
                                        convertir_a_meses(foto_mes, FirstMonth), 
                                        NA)]

  #valvula de seguridad para evitar valores infinitos
  #paso los infinitos a NULOS
  # Primero identifico columnas numéricas
  cols_numericas <- names(dataset)[sapply(dataset, function(x) is.numeric(x) && !is.list(x))]
  
  infinitos      <- lapply(cols_numericas, function(.name) {
    tryCatch({
      dataset[, sum(is.infinite(get(.name)))]
    }, error = function(e) {
      cat("Error en columna", .name, ":", e$message, "\n")
      0
    })
  })
  
  infinitos_qty  <- sum( unlist( infinitos) )
  if( infinitos_qty > 0 )
  {
    cat( "ATENCION, hay", infinitos_qty, "valores infinitos en tu dataset. Seran pasados a NA\n" )
    # Solo aplicar a columnas numéricas
    for(col in cols_numericas) {
      if(any(is.infinite(dataset[[col]]), na.rm = TRUE)) {
        dataset[[col]][is.infinite(dataset[[col]])] <- NA
      }
    }
  }


  #valvula de seguridad para evitar valores NaN  que es 0/0
  #paso los NaN a 0 , decision polemica si las hay
  #se invita a asignar un valor razonable segun la semantica del campo creado
  nans      <- lapply(cols_numericas, function(.name) {
    tryCatch({
      dataset[, sum(is.nan(get(.name)))]
    }, error = function(e) {
      cat("Error en columna", .name, ":", e$message, "\n")
      0
    })
  })
  
  nans_qty  <- sum( unlist( nans) )
  if( nans_qty > 0 )
  {
    cat( "ATENCION, hay", nans_qty, "valores NaN 0/0 en tu dataset. Seran pasados arbitrariamente a 0\n" )
    cat( "Si no te gusta la decision, modifica a gusto el programa!\n\n")
    # Solo aplicar a columnas numéricas
    for(col in cols_numericas) {
      if(any(is.nan(dataset[[col]]), na.rm = TRUE)) {
        dataset[[col]][is.nan(dataset[[col]])] <- NA
      }
    }
  }

  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#esta funcion supone que dataset esta ordenado por   <ticker, foto_q>
#calcula el lag y el delta lag
# ADAPTADO para financial data

Lags  <- function( cols, nlag, deltas )
{
  gc()
  sufijo  <- paste0( "_lag", nlag )

  dataset[ , paste0( cols, sufijo) := shift(.SD, nlag, NA, "lag"), 
             by= ticker, 
             .SDcols= cols]

  #agrego los deltas de los lags, con un "for" nada elegante
  if( deltas )
  {
    sufijodelta  <- paste0( "_delta", nlag )

    for( vcol in cols )
    {
     dataset[,  paste0(vcol, sufijodelta) := get( vcol)  - get(paste0( vcol, sufijo))]
    }
  }

  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#se calculan para los N trimestres previos el minimo, maximo y tendencia calculada con cuadrados minimos
#MANTIENE EXACTAMENTE la misma función C++ pero adaptada para quarters

cppFunction('NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde ) 
{
  /* Aqui se cargan los valores para la regresion */
  double  x[100] ;
  double  y[100] ;

  int n = pcolumna.size();
  NumericVector out( 5*n );

  for(int i = 0; i < n; i++)
  {
    //lag
    if( pdesde[i]-1 < i )  out[ i + 4*n ]  =  pcolumna[i-1] ;
    else                   out[ i + 4*n ]  =  NA_REAL ;


    int  libre    = 0 ;
    int  xvalor   = 1 ;

    for( int j= pdesde[i]-1;  j<=i; j++ )
    {
       double a = pcolumna[j] ;

       if( !R_IsNA( a ) ) 
       {
          y[ libre ]= a ;
          x[ libre ]= xvalor ;
          libre++ ;
       }

       xvalor++ ;
    }

    /* Si hay al menos dos valores */
    if( libre > 1 )
    {
      double  xsum  = x[0] ;
      double  ysum  = y[0] ;
      double  xysum = xsum * ysum ;
      double  xxsum = xsum * xsum ;
      double  vmin  = y[0] ;
      double  vmax  = y[0] ;

      for( int h=1; h<libre; h++)
      { 
        xsum  += x[h] ;
        ysum  += y[h] ; 
        xysum += x[h]*y[h] ;
        xxsum += x[h]*x[h] ;

        if( y[h] < vmin )  vmin = y[h] ;
        if( y[h] > vmax )  vmax = y[h] ;
      }

      out[ i ]  =  (libre*xysum - xsum*ysum)/(libre*xxsum -xsum*xsum) ;
      out[ i + n ]    =  vmin ;
      out[ i + 2*n ]  =  vmax ;
      out[ i + 3*n ]  =  ysum / libre ;
    }
    else
    {
      out[ i       ]  =  NA_REAL ; 
      out[ i + n   ]  =  NA_REAL ;
      out[ i + 2*n ]  =  NA_REAL ;
      out[ i + 3*n ]  =  NA_REAL ;
    }
  }

  return  out;
}')

#------------------------------------------------------------------------------
#calcula la tendencia de las variables cols de los ultimos N trimestres
#ADAPTADO para financial data

TendenciaYmuchomas  <- function( dataset, cols, ventana=6, tendencia=TRUE, minimo=TRUE, maximo=TRUE, promedio=TRUE, 
                                 ratioavg=FALSE, ratiomax=FALSE){
  gc()
  #Esta es la cantidad de trimestres que utilizo para la historia
  ventana_regresion  <- ventana

  last  <- nrow( dataset )

  #creo el vector_desde que indica cada ventana
  #de esta forma se acelera el procesamiento ya que lo hago una sola vez
  vector_ids   <- dataset$ticker

  vector_desde  <- seq( -ventana_regresion+2,  nrow(dataset)-ventana_regresion+1 )
  vector_desde[ 1:ventana_regresion ]  <-  1

  for( i in 2:last )  if( vector_ids[ i-1 ] !=  vector_ids[ i ] ) {  vector_desde[i] <-  i }
  for( i in 2:last )  if( vector_desde[i] < vector_desde[i-1] )  {  vector_desde[i] <-  vector_desde[i-1] }

  for(  campo  in   cols )
  {
    nueva_col     <- fhistC( dataset[ , get(campo) ], vector_desde ) 

    if(tendencia)  dataset[ , paste0( campo, "_tend", ventana) := nueva_col[ (0*last +1):(1*last) ]  ]
    if(minimo)     dataset[ , paste0( campo, "_min", ventana)  := nueva_col[ (1*last +1):(2*last) ]  ]
    if(maximo)     dataset[ , paste0( campo, "_max", ventana)  := nueva_col[ (2*last +1):(3*last) ]  ]
    if(promedio)   dataset[ , paste0( campo, "_avg", ventana)  := nueva_col[ (3*last +1):(4*last) ]  ]
    if(ratioavg)   dataset[ , paste0( campo, "_ratioavg", ventana)  := get(campo) /nueva_col[ (3*last +1):(4*last) ]  ]
    if(ratiomax)   dataset[ , paste0( campo, "_ratiomax", ventana)  := get(campo) /nueva_col[ (2*last +1):(3*last) ]  ]
  }
ReportarCampos(dataset)
}
#------------------------------------------------------------------------------
#Autor: Antonio Velazquez Bustamente,  UBA 2021
# ADAPTADO para financial data

Tony  <- function( cols )
{

  sufijo  <- paste0( "_tony")

  dataset[ , paste0( cols, sufijo) := lapply( .SD,  function(x){ x/mean(x, na.rm=TRUE)} ), 
             by= foto_mes, 
             .SDcols= cols]

  ReportarCampos( dataset )
}

#------------------------------------------------------------------------------
#Elimina del dataset las variables que estan por debajo de la capa geologica de canaritos
#ADAPTADO para financial data

GVEZ <- 1 

CanaritosImportancia  <- function( canaritos_ratio=0.2 )
{
  gc()
  ReportarCampos( dataset )
  
  canaritos_mes_end <- PARAMS$feature_engineering$const$canaritos_mes_end
  
  for( i  in 1:(ncol(dataset)*canaritos_ratio))  dataset[ , paste0("canarito", i ) :=  runif( nrow(dataset))]
  campos_buenos  <- setdiff( colnames(dataset), c("ticker","foto_mes",PARAMS$feature_engineering$const$clase ) )
  azar  <- runif( nrow(dataset) )
  dataset[ , entrenamiento := foto_mes>= PARAMS$feature_engineering$const$canaritos_mes_start &  foto_mes< canaritos_mes_end & azar < 0.10  ]
  
  dtrain  <- lgb.Dataset( data=    data.matrix(  dataset[ entrenamiento==TRUE, campos_buenos, with=FALSE]),
                          label=   dataset[ entrenamiento==TRUE, get(PARAMS$feature_engineering$const$clase)],
                          free_raw_data= FALSE
  )
  
  canaritos_mes_valid <- PARAMS$feature_engineering$const$canaritos_mes_valid
  
  dvalid  <- lgb.Dataset( data=    data.matrix(  dataset[ foto_mes==canaritos_mes_valid, campos_buenos, with=FALSE]),
                          label=   dataset[ foto_mes==canaritos_mes_valid, get(PARAMS$feature_engineering$const$clase)],
                          free_raw_data= FALSE
  )
  
  param <- list( objective= "regression",
                 metric= "rmse",
                 first_metric_only= TRUE,
                 boost_from_average= TRUE,
                 feature_pre_filter= FALSE,
                 verbosity= -100,
                 seed= 999983,
                 max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
                 min_gain_to_split= 0.0, #por ahora, lo dejo fijo
                 lambda_l1= 0.0,         #por ahora, lo dejo fijo
                 lambda_l2= 0.0,         #por ahora, lo dejo fijo
                 max_bin= 1023,            #por ahora, lo dejo fijo
                 num_iterations= 500,   #un numero muy grande, lo limita early_stopping_rounds
                 force_row_wise= TRUE,    #para que los alumnos no se atemoricen con tantos warning
                 learning_rate= 0.065, 
                 feature_fraction= 1.0,   #lo seteo en 1 para que las primeras variables del dataset no se vean opacadas
                 min_data_in_leaf= 260,
                 num_leaves= 60,
                 # num_threads= 8,
                 early_stopping_rounds= 50 )
  
  modelo  <- lgb.train( data= dtrain,
                        valids= list( valid= dvalid ),
                        # eval= fganancia_lgbm_meseta,
                        param= param,
                        verbose= -100 )
  
  tb_importancia  <- lgb.importance( model= modelo )
  tb_importancia[  , pos := .I ]
  
  GVEZ  <<- GVEZ + 1
  
  umbral  <- tb_importancia[ Feature %like% "canarito", median(pos) + 2*sd(pos) ] 
  
  col_utiles  <- tb_importancia[ pos < umbral & !( Feature %like% "canarito"),  Feature ]
  col_utiles  <-  unique( c( col_utiles,  c("ticker","foto_mes",PARAMS$feature_engineering$const$clase,"quarter_cycle","month_cycle", "foto_mes_reporte", "retraso_meses", "retraso_categoria") ) )
  col_inutiles  <- setdiff( colnames(dataset), col_utiles )
  
  dataset[  ,  (col_inutiles) := NULL ]
  
  
  ReportarCampos( dataset )
}
#------------------------------------------------------------------------------
#agrega para cada columna de cols una nueva variable _rank  que es un numero entre 0 y 1  del ranking de esa variable ese trimestre
# ADAPTADO para financial data

Rankeador  <- function( cols )
{
  gc()
  sufijo  <- "_rank" 

  for( vcol in cols )
  {
     dataset[ , paste0( vcol, sufijo) := frank( get(vcol), ties.method= "random")/ .N, 
                by= foto_mes ]
  }

  ReportarCampos( dataset )
}

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
############## Aqui empieza el programa ################

#cargo el dataset
setwd(PARAMS$environment$base_dir)
setwd(PARAMS$environment$data_dir)
nom_arch  <-  PARAMS$feature_engineering$files$input$dentrada
dataset   <- fread( nom_arch )

#ordeno el dataset por <ticker, foto_q> para poder hacer lags
setorderv( dataset, PARAMS$feature_engineering$const$campos_sort )

###################################################################

# Crear la clase intermedia - ADAPTADO para financial data
dataset[,PARAMS$feature_engineering$const$clase:=
  get(PARAMS$feature_engineering$const$origen_clase)
,by=ticker
]
# Aca hago el lead segun cuanto quiera predecir hacia el futuro
dataset[,PARAMS$feature_engineering$const$clase:=shift(
  get(PARAMS$feature_engineering$const$clase)
  ,n=PARAMS$feature_engineering$const$orden_lead
  ,type = "lead"
  ),by=ticker]


AgregarTrimestre( dataset )  #agrego el ciclo del trimestre
AgregarMes( dataset )  #agrego el ciclo del mes

if( length( PARAMS$feature_engineering$param$variablesdrift) > 0 )    DriftEliminar( dataset, PARAMS$feature_engineering$param$variablesdrift )

if( PARAMS$feature_engineering$param$dummiesNA )  DummiesNA( dataset )  #esta linea debe ir ANTES de Corregir  !!

if( PARAMS$feature_engineering$param$corregir == "ClaudioCastillo" )  CorregirClaudioCastillo( dataset )  #esta linea debe ir DESPUES de  DummiesNA
if( PARAMS$feature_engineering$param$corregir == "AsignarNA" )       CorregirNA( dataset )  #esta linea debe ir DESPUES de  DummiesNA

if( PARAMS$feature_engineering$param$variablesmanuales )  AgregarVariables( dataset )

dataset[,PARAMS$feature_engineering$const$origen_clase:=NULL]
#--------------------------------------
#Esta primera parte es muuuy  artesanal  y discutible  ya que hay multiples formas de hacerlo

cols_lagueables  <- copy( setdiff( colnames(dataset), PARAMS$feature_engineering$const$campos_fijos ) )


for( i in 1:length( PARAMS$feature_engineering$param$tendenciaYmuchomas$correr ) ){
  if( PARAMS$feature_engineering$param$tendenciaYmuchomas$correr[i] ) 
  {
    #veo si tengo que ir agregando variables
    if( PARAMS$feature_engineering$param$acumulavars )  cols_lagueables  <- setdiff( colnames(dataset), PARAMS$feature_engineering$const$campos_fijos )

    cols_lagueables  <- intersect( colnames(dataset), cols_lagueables )
    TendenciaYmuchomas( dataset, 
                        cols= cols_lagueables,
                        ventana=   PARAMS$feature_engineering$param$tendenciaYmuchomas$ventana[i],
                        tendencia= PARAMS$feature_engineering$param$tendenciaYmuchomas$tendencia[i],
                        minimo=    PARAMS$feature_engineering$param$tendenciaYmuchomas$minimo[i],
                        maximo=    PARAMS$feature_engineering$param$tendenciaYmuchomas$maximo[i],
                        promedio=  PARAMS$feature_engineering$param$tendenciaYmuchomas$promedio[i],
                        ratioavg=  PARAMS$feature_engineering$param$tendenciaYmuchomas$ratioavg[i],
                        ratiomax=  PARAMS$feature_engineering$param$tendenciaYmuchomas$ratiomax[i]
                      )
  # elimino las variables poco importantes, para hacer lugar a las importantes
    if( PARAMS$feature_engineering$param$tendenciaYmuchomas$canaritos[ i ] > 0 )  CanaritosImportancia( canaritos_ratio= unlist(PARAMS$feature_engineering$param$tendenciaYmuchomas$canaritos[ i ]) )

  }
}


for( i in 1:length( PARAMS$feature_engineering$param$lags$correr ) ){
  if( PARAMS$feature_engineering$param$lags$correr[i] )
  {
    #veo si tengo que ir agregando variables
    if( PARAMS$feature_engineering$param$acumulavars )  cols_lagueables  <- setdiff( colnames(dataset), PARAMS$feature_engineering$const$campos_fijos )

    cols_lagueables  <- intersect( colnames(dataset), cols_lagueables )
    Lags( cols_lagueables, 
          PARAMS$feature_engineering$param$lags$lag[i], 
          PARAMS$feature_engineering$param$lags$delta[ i ] )   #calculo los lags de orden  i

    #elimino las variables poco importantes, para hacer lugar a las importantes
    if( PARAMS$feature_engineering$param$lags$canaritos[ i ] > 0 )  CanaritosImportancia( canaritos_ratio= unlist(PARAMS$feature_engineering$param$lags$canaritos[ i ]) )
  }
}


if( PARAMS$feature_engineering$param$acumulavars )  cols_lagueables  <- setdiff( colnames(dataset), PARAMS$feature_engineering$const$campos_fijos )

#agrego los rankings
if( PARAMS$feature_engineering$param$rankeador ) {
  if( PARAMS$feature_engineering$param$acumulavars )  cols_lagueables  <- setdiff( colnames(dataset), PARAMS$feature_engineering$const$campos_fijos )

  cols_lagueables  <- intersect( colnames(dataset), cols_lagueables )
  setorderv( dataset, PARAMS$feature_engineering$const$campos_rsort )
  Rankeador( cols_lagueables )
  setorderv( dataset, PARAMS$feature_engineering$const$campos_sort )
}

if( PARAMS$feature_engineering$param$canaritos_final > 0  )   CanaritosImportancia( canaritos_ratio= PARAMS$feature_engineering$param$canaritos_final )

#dejo la clase como ultimo campo
nuevo_orden  <- c( setdiff( colnames( dataset ) , PARAMS$feature_engineering$const$clase ) , PARAMS$feature_engineering$const$clase )
setcolorder( dataset, nuevo_orden )

# Corrijo los NANs que surgieron de los ratios
cols_lagueables  <- copy( setdiff( colnames(dataset), PARAMS$feature_engineering$const$campos_fijos ) )
for(col in cols_lagueables){
  dataset[[col]][is.nan(dataset[[col]])] <- 0
}

# Filtro los datos: solo observaciones con clase definida O el mes presente
dataset <- dataset[(!is.na(get(PARAMS$feature_engineering$const$clase)))|foto_mes>=PARAMS$feature_engineering$const$presente]

#Grabo el dataset
experiment_dir <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,sep = "_")
experiment_lead_dir <- paste(PARAMS$experiment$experiment_label,PARAMS$experiment$experiment_code,paste0("f",PARAMS$feature_engineering$const$orden_lead),sep = "_")

setwd(PARAMS$environment$base_dir)
setwd("./exp")

dir.create(experiment_dir,showWarnings = FALSE)
setwd(experiment_dir)
dir.create(experiment_lead_dir,showWarnings = FALSE)
setwd(experiment_lead_dir)

PARAMS$features$features_n <- length(colnames(dataset))
PARAMS$features$colnames <- colnames(dataset)

jsontest = jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE)
sink(file = paste0(experiment_lead_dir,".json"))
print(jsontest)
sink(file = NULL)

dir.create("01_FE",showWarnings = FALSE)
setwd("01_FE")

fwrite( dataset,
        paste0( experiment_lead_dir,".csv.gz" ),
        logical01= TRUE,
        sep= "," )

cat("\n=== FEATURE ENGINEERING FINANCIAL FORECASTING COMPLETADO ===\n")
cat("Dimensiones finales del dataset:", nrow(dataset), "x", ncol(dataset), "\n")
cat("Archivo guardado exitosamente\n")
