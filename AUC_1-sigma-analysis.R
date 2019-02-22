# path = "/home/antonio/PycharmProjects/Deep_Learning_for_Astrophysics/results/MODELING/CLASSIFICATION/"
source('Utils.R')
load('data.RData')

table( data$Model, data$Method)[ , 2:9 ]
cols = c("Method", "Model", "n_variables", "Accuracy", 'AUC')
data = data[ , cols ]

### ++++++++++ Best results for each Model ++++++++++ ### 

best = data %>% group_by( Model ) %>% filter( AUC == max( AUC ))
best = best %>% group_by( Model ) %>% filter( n_variables == min( n_variables ))
best = best[!duplicated( best$Model, best$n_variables, best$AUC, best$AUC),]

write.csv2( best, file = paste0( dir_result, '/best_AUC.csv') )

### +++++++++++++++++++++++++++++++++++++++++++++++++ ###


### ++++++++++ Isolation of ISIS ++++++++++ ### 

data_ISIS = data[ data$Method == 'ISIS',]
best_ISIS = data_ISIS %>% group_by( Model ) %>% filter( AUC == max( AUC ))
best_ISIS = best_ISIS %>% group_by( Model ) %>% filter( n_variables == min( n_variables ))
best_ISIS = best_ISIS[!duplicated( best_ISIS$Model, best$n_variables, best_ISIS$AUC, best_ISIS$AUC),]

write.csv2( best, file = paste0( dir_result, '/best_ISIS_AUC.csv') )

### +++++++++++++++++++++++++++++++++++++++++++++++++ ###


data$model_method = paste0( data$Model, '_', data$Method )

avg = as.data.frame( tapply( X = data$AUC, INDEX = data$model_method, FUN = mean) )
colnames( avg ) = 'avg'
avg$model_method = row.names( avg )
avg$sd = tapply( X = data$AUC, INDEX = data$model_method, FUN = sd)

avg = as.data.frame( avg )
data = merge( data, avg )

data$upper = data$avg + data$sd
data$lower = data$avg - data$sd

data$FLG = ifelse( test = ( data$AUC >= data$lower & data$AUC <= data$upper), yes = T, no = F)
dim( data[ data$FLG == T, ])

df_one_sigma = data.frame( Max = tapply(data$n_variables[ data$FLG == T], INDEX = data$model_method[ data$FLG == T], FUN = max),
                           Min = tapply(data$n_variables[ data$FLG == T], INDEX = data$model_method[ data$FLG == T], FUN = min),
                           AUC = tapply( X = data$AUC, INDEX = data$model_method, FUN = mean), 
                           std_dev = tapply( X = data$AUC, INDEX = data$model_method, FUN = sd),
                           AUC_max = tapply(data$AUC, INDEX = data$model_method, FUN = max))


df_one_sigma$Model = NA
df_one_sigma$Method = NA
df_one_sigma$exact_AUC = NA


for ( model in unique( data$Model) )
  for (method in unique( data$Method ))
  {
    print( c(model, method ))
    # model = data$Model[ 5] 
    # method = data$Method[ 6 ]
    ix = grep( paste0('^', model, '_', method), row.names( df_one_sigma ))
    df_one_sigma[ ix , c("Model", "Method") ] = c( model, method)
    AUC = data[ data$Method == method & data$Model == model & data$n_variables == df_one_sigma[ ix, 'Min'], 'AUC']
    df_one_sigma[ ix, 'exact_AUC'] = AUC
    
  }

df_one_sigma = data.frame( df_one_sigma)
df_one_sigma$upper = df_one_sigma$AUC + df_one_sigma$std_dev
df_one_sigma$lower = df_one_sigma$AUC - df_one_sigma$std_dev
df_one_sigma$range = df_one_sigma$upper- df_one_sigma$lower

df_one_sigma$FLG_MAX = ifelse( test = ( df_one_sigma$AUC_max >= df_one_sigma$lower & df_one_sigma$AUC_max <= df_one_sigma$upper), yes = T, no = F)

table(df_one_sigma$FLG_MAX)

# View( df_one_sigma[ df_one_sigma$FLG_MAX == F ,])
# View( df_one_sigma[ df_one_sigma$FLG_MAX == T ,])
# View( data[ data$Model == 'GBM' ,])
####################################################################################

tapply( df_one_sigma$AUC, df_one_sigma$Model, FUN = max)
df_one_sigma = df_one_sigma[ df_one_sigma$FLG_MAX == T ,]

### ++++++++++ Best results for each Model ++++++++++ ### 

best_IC = df_one_sigma %>% group_by( Model ) %>% filter( AUC == max( AUC ))

write.csv2( best, file = paste0( dir_result, '/best_IC.csv') )
### +++++++++++++++++++++++++++++++++++++++++++++++++ ###

####################################################################################


# data_KW = as.data.frame( df_one_sigma[ , c('Model', 'Method', 'AUC')] )
# data_KW$Method = as.factor( data_KW$Method )
# row.names( data_KW ) = 1: length( row.names( data_KW ))
# 
# friedman.test(formula = AUC ~ Method|Model, data = data_KW)
# 
# alpha_corrected = 0.1/28
# 
# # for( col in 1:ncol( unique(data$Method) ) )
# # {
# #   data_KW[ col ]
# # }
# 
# multi_test = expand.grid( Group1 = unique(data$Method), Group2 = unique(data$Method) )
# multi_test$p_value = NA
# 
# for ( row in 1:nrow( multi_test))
# {
#   # row = 2
#   g1 = data_KW$AUC[ data_KW$Method == as.character( multi_test[ row , 'Group1']) ] 
#   g2 = data_KW$AUC[ data_KW$Method == as.character( multi_test[ row , 'Group2']) ] 
#   multi_test[ row , 'p_value'] = t.test(  g1, g2, paired = T)$p.value
# }
# 
# 
# # data_KW = reshape( prova, idvar =  'Model', timevar = 'Method', direction = 'wide')
# 
# row.names( data_KW ) = 1: length( row.names( data_KW ))
# colnames( data_KW ) = gsub( 'exact_AUC.', '', colnames( data_KW ))
# 
# df_one_sigma = df_one_sigma[ df_one_sigma$Method != 'ISIS', ]
# 
# 
# 
# tab = tapply(df_one_sigma$std_dev, INDEX = df_one_sigma$Model, FUN = min)
# df = data.frame( Model = names(tab), Min_std = tab)
# df_one_sigma_v2 = df_one_sigma[ 1==0,]
# 
# 
# for( i in 1:nrow( df))
# {
#    row = df_one_sigma[ df_one_sigma$std_dev == df$Min_std[ i ] & df_one_sigma$Model == df$Model[ i ], ] 
#    df_one_sigma_v2 = rbind( df_one_sigma_v2, row)
# }


###############################################################



# library(grid)
# library(gridExtra)
# library(ggplot2)
# 
# path = "/home/antonio/PycharmProjects/Deep_Learning_for_Astrophysics/results/MODELING/CLASSIFICATION/"
# data = read.csv( paste0(path, "metrics.csv") )
# data_NN = read.csv( paste0(path, 'NEURAL_NETWORK/metrics.csv'))
# 
# data = rbind(data, data_NN)
# 
# data = data[ data$Treshold == 0.5, ]
# data = data[ data$Method != 'DECISION_TREE', ]
# 
# 
# 
# data = data[order( data$Model, data$Method, data$n_variables ), ]
# data$model_method = paste0( data$Model, '_', data$Method )
# 
# avg = as.data.frame( tapply( X = data$AUC, INDEX = data$Model, FUN = mean) )
# colnames( avg ) = 'avg'
# avg$Model = row.names( avg )
# avg$sd = tapply( X = data$AUC, INDEX = data$Model, FUN = sd)
# 
# data = merge( data, avg )
# 
# data$upper = data$avg + data$sd
# data$lower = data$avg - data$sd
# 
# data$FLG = ifelse( test = ( data$AUC >= data$lower & data$AUC <= data$upper), yes = T, no = F)
# dim( data[ data$FLG == T, ])
# 
# 
# df_one_sigma = data.frame( Max = tapply(data$n_variables[ data$FLG == T], INDEX = data$Model[ data$FLG == T], FUN = max),
#                            Min = tapply(data$n_variables[ data$FLG == T], INDEX = data$Model[ data$FLG == T], FUN = min),
#                            AUC = tapply( X = data$AUC, INDEX = data$Model, FUN = mean), 
#                            std_dev = tapply( X = data$AUC, INDEX = data$Model, FUN = sd) )
# 
# data_temp = data[]
# 
# 
