exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 1

# exec(open("015_SPLITTING_DATA.py").read(), globals())
# exec(open("030_VARIABLES_SELECTION.py").read(), globals())
# exec(open("035_UNIVARIATE_VARIABLES_SELECTION.py").read(), globals())

nvars = np.arange(140, 252, 20) #np.concatenate( ([1], np.arange(10, 51, 10), np.arange(70, 140, 30)) )
methods = ['LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']


# predictors = extract_predictors( method, nvar, SEED)
# eff_nvar = len(predictors)
probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()

scheduled_model = 'running_model/'
create_dir( scheduled_model)

for method in methods:
    for nvar in nvars:
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, eff_nvar
        try:
            exec(open("041_TREE_BASED_MODELS.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_TREE_BASED_MODELS' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv( scheduled_model + 'ERROR_TREE_BASED_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("043_REGULARIZED_METHODS.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_REGULARIZED_METHODS' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv( scheduled_model + 'ERROR_LASSO_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("045_NAIVE_BAYES.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_NAIVE_BAYES' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv( scheduled_model + 'ERROR_NAIVE_BAYES_' + method + '_' + str(nvar) + '.csv')
        try:
            exec(open("046_KNN.py").read(), globals())
            DF.to_csv( scheduled_model + 'OK_KNN_' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv(  scheduled_model + 'ERROR_' + method + '_' + str(nvar) + '.csv' )
        try:
            exec (open("042_SVM.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_SVM' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv( scheduled_model + 'ERROR_SVM_' + method + '_' + str(nvar) + '.csv')


for method in methods:
    for nvar in nvars:
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, nvar
    try:
        exec(open("051_NEURAL_NETWORK.py").read(), globals())
        DF.to_csv(scheduled_model + 'OK_NN_' + method + '_' + str(nvar) + '.csv')
    except:
        DF.to_csv( scheduled_model + 'ERROR_NN_' + method + '_' + str(nvar) + '.csv')









#########################################################################################
######## ****** RANDOM FOREST ****** ####################################################
################################### ****** E ****** #####################################
#################################################### ****** GBM  ****** #################
#########################################################################################

# srun -N 1 -n16 -A cin_staff -t300  -p gll_usr_gpuprod --gres=gpu:kepler:2 --pty /bin/bash
# module load python/2.7.12
# source py2/bin/activate
# cd INAF/
# python


# exec(open("Utils.py").read(), globals())
# exec(open("000_Utils_parallel.py").read(), globals())
#
# SEED = 741
# njob = 2
#
#
#
# nvars = np.concatenate( ([1], np.arange(10, 51, 10), np.arange(70, 140, 30)) )
# methods = ['ISIS', 'LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']
#
#
# probs_to_check = np.arange(0.1, 0.91, 0.1)
# DF = pd.DataFrame()
#
# scheduled_model = 'running_model/'
# create_dir( scheduled_model)
#
# for method in methods:
#     for nvar in nvars:
#         # nvar = 100
#         predictors = extract_predictors(method, nvar, SEED)
#         eff_nvar = len(predictors)
#         print method, nvar
#         try:
#             exec(open("000_GALILEO_TREE_BASED.py").read(), globals())
#             DF.to_csv(scheduled_model + 'OK_RF_GBM_' + method + '_' + str(nvar) + '.csv')
#         except:
#             DF.to_csv( scheduled_model + 'ERROR_RF_GBM_' + method + '_' + str(nvar) + '.csv')
#
#
#
#
#
#
#
#
# #########################################################################################
# ########### ****** SUPPORT ****** #######################################################
# ############################### ****** VECTOR ****** ####################################
# ################################################# ****** MACHINE ****** #################
# #########################################################################################
#
# # srun -N 1 -n16 -A cin_staff -t300  -p gll_usr_gpuprod --gres=gpu:kepler:2 --pty /bin/bash
# # module load python/2.7.12
# # source py2/bin/activate
# # cd INAF/
# # python
#
# exec(open("Utils.py").read(), globals())
# exec(open("Utils_parallel.py").read(), globals())
#
# SEED = 741
# njob = 16
#
# # exec(open("015_SPLITTING_DATA.py").read(), globals())
# # exec(open("030_VARIABLES_SELECTION.py").read(), globals())
# # exec(open("035_UNIVARIATE_VARIABLES_SELECTION.py").read(), globals())
#
# nvars = np.concatenate( ([1], np.arange(10, 51, 10), np.arange(70, 140, 30)) )
# methods = ['ISIS', 'LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']
#
#
# # predictors = extract_predictors( method, nvar, SEED)
# # eff_nvar = len(predictors)
# probs_to_check = np.arange(0.1, 0.91, 0.1)
# DF = pd.DataFrame()
#
# scheduled_model = 'running_model/'
# create_dir( scheduled_model)
#
#
# nvars = [1, 3, 5]
# methods = ['LR_ACCURACY']
#
# for method in methods:
#     for nvar in nvars:
#         predictors = extract_predictors(method, nvar, SEED)
#         eff_nvar = len(predictors)
#         print method, nvar
#         try:
#             exec (open("042_SVM.py").read(), globals())
#             DF.to_csv(scheduled_model + 'OK_SVM' + method + '_' + str(nvar) + '.csv')
#         except:
#             DF.to_csv( scheduled_model + 'ERROR_SVM_' + method + '_' + str(nvar) + '.csv')
#
#
