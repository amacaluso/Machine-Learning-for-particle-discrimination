exec(open("Utils.py").read(), globals())
exec(open("0100_Reg_pre_processing.py").read(), globals())


RANDOM_SEED = 300

data.drop(data.columns[0], axis=1)

training_set, test_set = train_test_split( data, test_size = 0.2,
                                           random_state = RANDOM_SEED)


cols_to_remove = [u'index', u'FILE', u'TTree', u'TIME', u'PID', u'EVENT_NUMBER',
                  u'EVENT_TYPE', u'DIRNAME', u'FLG_BRNAME01', u'FLG_EVSTATUS', u'Y' ]


training_set = training_set.drop( cols_to_remove, axis = 1 )
test_set = test_set.drop( cols_to_remove, axis=1 )

target_variable = 'Y'

X = training_set.drop( target_variable, axis = 1).astype( np.float32 )
Y = training_set[ target_variable ]

X_test = test_set.drop( target_variable, axis = 1).astype( np.float32 )
Y_test = test_set[ target_variable ]

x_names = X.columns