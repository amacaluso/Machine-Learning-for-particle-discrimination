ensureLibrary = function(packageName) {
  
  isPresent = any(row.names(installed.packages()) == packageName)
  if(! isPresent) {
    install.packages(packageName)
  }
  
  library(packageName, character.only=TRUE)
}



remove_columns_by_names = function (df, colNames)
{
  colnames_to_keep = colnames(df)[ ! colnames(df) %in% colNames ]
  df = as.data.frame( df )[ , colnames_to_keep ]
  df
}


ROC_analysis = function(
  prediction, 
  y_true, 
  probability_thresholds =  seq( 0.2 , 0.8, by = 0.05 ))
{  
  matrice_risultati = lapply( probability_thresholds, function( tresh, prediction, y_true  ) {
    
    y_hat = as.numeric( prediction >= tresh )
    
    roc_row = list(
      y_1__y_hat_1 = sum( y_true == 1 & y_hat == 1  ),
      y_1__y_hat_0 = sum( y_true == 1 & y_hat == 0  ),
      y_0__y_hat_1 = sum( y_true == 0 & y_hat == 1  ),
      y_0__y_hat_0 = sum( y_true == 0 & y_hat == 0  )
    )
    
    roc_row
    
  }, prediction, y_true ) 
  
  df_roc = rbindlist( matrice_risultati )
  
  df_roc$accuracy = ( df_roc$y_1__y_hat_1 + df_roc$y_0__y_hat_0 ) / length(y_true)
  df_roc$specificity = df_roc$y_0__y_hat_0 / sum( y_true == 0 )
  df_roc$sensitivity__recall = df_roc$y_1__y_hat_1 / sum( y_true == 1 )
  df_roc$positive_predictive_value = df_roc$y_1__y_hat_1 / ( df_roc$y_1__y_hat_1 + df_roc$y_0__y_hat_1 )
  
  fpr = df_roc$y_0__y_hat_1 / ( df_roc$y_0__y_hat_1 + df_roc$y_0__y_hat_0 ) 
  df_roc$positive_likelihood_ratio = df_roc$sensitivity / fpr
  
  df_roc$F1_score = 2 * ( ( df_roc$sensitivity__recall * df_roc$positive_predictive_value )
                          / ( df_roc$sensitivity__recall + df_roc$positive_predictive_value ) )
  
  colnames( df_roc ) <- c(
    "(y=1,y_hat=1) TP",
    "(y=1,y_hat=0) FN",
    "(y=0,y_hat=1) FP",
    "(y=0,y_hat=0) TN",
    "Accuracy: true/total",
    "Specificity: TN/negative",
    "Sensitivity (AKA Recall): TP/positive",
    "Positive Predictive Value: TP/predicted_positive",
    "Positive Likelihood Ratio",
    "F1_SCORE"
  )
  
  df_roc = cbind( probability_thresholds, df_roc ) 
  #rownames( df_roc ) <- probability_thresholds
  
  df_roc
}




multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) 
{
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}




CV<-function(x){
  sd(x)/abs(mean(x))
}




