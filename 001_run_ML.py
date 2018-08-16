exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 231
# exec(open("015_SPLITTING_DATA.py").read(), globals())
# exec(open("030_VARIABLES_SELECTION.py").read(), globals())


njob = 1
method = 'ISIS'
# GET PREDICTOR
# ['LASSO', 'DECISION_TREE', 'RANDOM_FOREST', 'GBM',
#  'E_NET', 'INFORMATION_GAIN', 'LR_ACCURACY']
# ISIS

nvar = 10
probs_to_check = np.arange(0.1, 0.91, 0.1)

# exec(open("041_TREE_BASED_MODELS.py").read(), globals())
# exec(open("042_SVM.py").read(), globals())
# exec(open("043_REGULARIZED_METHODS.py").read(), globals())
exec(open("045_NAIVE_BAYES.py").read(), globals())
exec(open("046_KNN.py").read(), globals())
