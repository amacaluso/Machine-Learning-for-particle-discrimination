
# ************* RF E GBM ***************** #
############################################

# srun -N 1 -n16 -A cin_staff -t300  -p gll_usr_gpuprod --gres=gpu:kepler:2 --pty /bin/bash
# module use /gpfs/scratch/userinternal/epascol1/spack/share/spack/modules/linux-centos7-x86_64
# module load  gcc-6.3.0-gcc-7.3.0-us4i5fv
# module load  cuda-9.0.176-gcc-6.3.0-xjorzpo
# module load  cudnn-7.0.5-gcc-6.3.0-ceoy3cj
# module load python/2.7.12
# source py2/bin/activate
# cd INAF/
# python



exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 16

# exec(open("015_SPLITTING_DATA.py").read(), globals())
# exec(open("030_VARIABLES_SELECTION.py").read(), globals())
# exec(open("035_UNIVARIATE_VARIABLES_SELECTION.py").read(), globals())

#method = 'ISIS'
# GET PREDICTOR ['ISIS', 'LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']


probs_to_check = np.arange(0.1, 0.91, 0.1)
DF = pd.DataFrame()

scheduled_model = 'running_model/'
create_dir( scheduled_model)


# 'ISIS', 'LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']
nvars = [ 110, 130 ]
methods = ['INFORMATION_GAIN']

for method in methods:
    for nvar in nvars:
        # nvar = 100
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, nvar
        try:
            exec (open("051_NEURAL_NETWORK.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_NN_' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv(scheduled_model + 'ERROR_NN_' + method + '_' + str(nvar) + '.csv')




# 'ISIS', 'LR_ACCURACY', 'E_NET', 'INFORMATION_GAIN', 'LASSO', 'RIDGE', 'RANDOM_FOREST', 'GBM']
nvars = [ 130 ]
methods = ['LR_ACCURACY']

for method in methods:
    for nvar in nvars:
        # nvar = 100
        predictors = extract_predictors(method, nvar, SEED)
        eff_nvar = len(predictors)
        print method, nvar
        try:
            exec (open("051_NEURAL_NETWORK.py").read(), globals())
            DF.to_csv(scheduled_model + 'OK_NN_' + method + '_' + str(nvar) + '.csv')
        except:
            DF.to_csv(scheduled_model + 'ERROR_NN_' + method + '_' + str(nvar) + '.csv')


