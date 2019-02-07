import seaborn as sns
sns.set(style="ticks")

dots = sns.load_dataset("dots")

# Define a palette to ensure that colors will be
# shared across the facets
palette = dict(zip(dots.coherence.unique(),
                   sns.color_palette("rocket_r", 6)))
data.columns
# Plot the lines on two facets
sns.relplot(x="n_variables", y="Accuracy",
            hue="Model",
            size_order=[['E_NET', 'GBM', 'INFORMATION_GAIN'],
                        ['ISIS', 'LASSO', 'LR_ACCURACY'],
                        ['RANDOM_FOREST', 'RIDGE']],
            # size="choice",
            col="Method",
            # size_order=["T1", "T2"],
            # palette=palette,
            height=5, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=data)
plt.show()

# Plot the lines on two facets
sns.relplot(x="time", y="firing_rate",
            hue="coherence", size="choice", col="align",
            size_order=[['E_NET', 'GBM', 'INFORMATION_GAIN'],
                        ['ISIS', 'LASSO', 'LR_ACCURACY'],
                        ['RANDOM_FOREST', 'RIDGE']],
            palette=palette,
            height=5, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=dots)
plt.show()









exec(open("Utils.py").read(), globals())
exec(open("Utils_parallel.py").read(), globals())

SEED = 741
njob = 20

# exec(open("015_SPLITTING_DATA.py").read(), globals())
# exec(open("030_VARIABLES_SELECTION.py").read(), globals())
# exec(open("035_UNIVARIATE_VARIABLES_SELECTION.py").read(), globals())

nvars = [200, 240, 220] #np.arange(160, 252, 20) #np.concatenate( ([1], np.arange(10, 51, 10), np.arange(70, 140, 30)) )
methods = ['E_NET']


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
        exec(open("041_RANDOM_FOREST.py").read(), globals())






