
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

exec(open("Utils.py").read(), globals())

import ROOT as R
import rootpy
# from rootpy import root2hdf5 as hdf
# import rootpy.io
# from rootpy.tree import Tree
from root_numpy import root2array, tree2array


file_kB2 = R.TFile.Open( "AGILE/TMVA_INPUT.kB2.root" )

labels_kB2 = ["P100", "ELE", "POS", "P200", "P50", "P500", "P1000",
              "P10000", "PHE", "PLE", "REA" ]


file_B2_photons = R.TFile.Open( "AGILE/TMVA_INPUT.B2.photons.root" )

labels_B2_photons = ["P25", "P39", "P59", "P84", "P141", "P283", "P632", 
                     "P1414", "P3162", "P7071", "P17320", "P40000", "P65",
                     "P250", "P700", "P2000", "P500", "P200", "P100", "P1000",
                     "P50", "P400" ]




complete_dataframe = pd.DataFrame()


files = [ file_kB2, file_B2_photons ]
TTree_labels =  [ labels_kB2, labels_B2_photons ]

# =============================================================================
# for i_file in range( len(files) ) : 
#     
#     file = files[ i_file ]
#     labels = TTree_labels[ i_file ]
#     print( file, "\n", labels, i_file )
# =============================================================================
for i in range( len(files) ):

    file = files[ i ]
    labels = TTree_labels[ i ]
    
    for label in labels:
        
        df = pd.DataFrame()
        print( label, "\n\n\n" )
        current_tree = file.Get( label )
        n_rows = current_tree.GetEntries()
        
        list_of_branches = current_tree.GetListOfBranches()
        n_var = list_of_branches.GetSize()
        list_of_var_name = []
        
        print( "tree --> ", label, "n_righe --> ", n_rows, "n_col --> ", n_var )
    
            
        lista = []
        
        for k in range( n_var ):
    
            current_name = list_of_branches[k].GetName()
            current_array =  tree2array(current_tree, current_name  )
            
            if( len( current_array.shape ) > 1 ):
                print( k, 
                      "tipo -> ", type(current_array[k]),
                      "name = ", list_of_branches[k].GetName(),
                      "dimension = ", current_array.shape, 
                      "tipo valori = ", current_array.dtype )
                
                for i in range( current_array.shape[1] ):
                    
                    new_array = np.array( range( current_array.shape[0]) )
                    for j in range( current_array.shape[0] ):
                        new_array[j] = current_array[j][i] 
    
                    lista.append( new_array )
                    list_of_var_name.append( list_of_branches[k].GetName() + "_" + str(i) )
                    
            else:
                list_of_var_name.append( list_of_branches[k].GetName() )
                lista.append( current_array )
        
        df = pd.DataFrame( lista )
        df = df.transpose()
        df.columns = list_of_var_name
        
        TTree = pd.Series( np.repeat( label, n_rows) )
        current_file = pd.Series( np.repeat( file.GetName(), n_rows) ) 
        
        
        df = pd.concat( [ current_file, TTree, df ], axis = 1 )  
        df.columns = [ 'FILE', 'TTree' ] + list_of_var_name
        
        complete_dataframe = complete_dataframe.append( df )
        print( label, "\n\n\n" )





groups = pd.crosstab(complete_dataframe.ix[ : , 0:2 ].FILE,
                     complete_dataframe.ix[ : , 0:2 ].TTree ).transpose()


for i in range( len(groups.index) ):
    if groups.ix[i, 0] == groups.ix[i,1]:
        complete_dataframe = complete_dataframe.ix[ -((complete_dataframe.FILE == groups.columns[1]) 
                                & (complete_dataframe.TTree == groups.index[i])), : ]


complete_dataframe.EVENT_TYPE.value_counts()
df = complete_dataframe.copy()    


nrows = complete_dataframe.shape[ 0 ]
Y = pd.Series( np.repeat( 0, nrows) ) 

df = pd.concat( [ df.reset_index(), Y ], axis = 1 )

Y_1 =[ 'G_100', 'G_1000', 'G_10000', 'G_141', 'G_1414', 'G_17320', 'G_200', 
       'G_2000', 'G_25', 'G_250', 'G_283', 'G_3162', 'G_39', 'G_400', 
       'G_40000', 'G_50','G_500', 'G_59', 'G_632', 'G_65', 'G_700', 
       'G_7071', 'G_84' ]


colnames = []
for col in df.columns[0:261]:
    colnames.append(col)

colnames.append( 'Y' )

df.columns = colnames


for i in range( nrows ):
    if df.ix[ i, 'EVENT_TYPE'].decode("utf-8") in Y_1:
        df.ix[ i, 'Y'] = 1

col_to_group = [6, 261]
col_to_group_II = [ 6, 261 ]


groups_Y = pd.crosstab(df.ix[ : , col_to_group].EVENT_TYPE, df.ix[:, col_to_group_II ].Y )
groups_Y.to_csv( "DATA/target_variable.csv")



EVENT_TYPE_DECODE = []

for i in range( nrows ):
    EVENT_TYPE_DECODE.append( df.ix[ i, 'EVENT_TYPE'].decode("utf-8") )

df['EVENT_TYPE_DECODE'] = EVENT_TYPE_DECODE

n_samples = np.sum( groups_Y.ix[:, 0] )

labels_Y0 = []

for label in groups_Y.index[ groups_Y.ix[:, 0] > 0 ]:
    labels_Y0.append( label.decode( "utf-8"))


df_Y_0 = df[ df.EVENT_TYPE_DECODE.isin (labels_Y0) ]
df_Y_1 = df[ df.EVENT_TYPE_DECODE.isin (Y_1) ]

balanced_df = pd.concat( [df_Y_1.sample( n = n_samples ), df_Y_0] )

nrows = balanced_df.shape[ 0 ]
balanced_df.columns

balanced_df.to_csv( "DATA/balanced_df.csv", index = False)





