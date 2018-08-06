exec(open("Utils.py").read(), globals())

dir_data = 'DATA/CLASSIFICATION/'
data = pd.read_csv( dir_data + "dataset.csv" )



try:
   SEED
except NameError:
    SEED = 123
    print 'SEED does not exists and will be set automatically to', SEED
else:
    print 'The SEED is', SEED

print 'The dimension of complete dataset is', data.shape

dir_dest = dir_data + str(SEED) +'/'
create_dir( dir_dest )

variable_sub_dataset, modeling_dataset = train_test_split( data, test_size = 0.9,
                                                           random_state = SEED)

variable_sub_dataset.to_csv( dir_dest + 'pre_training_set.csv', index = False)
#modeling_dataset.to_csv( dir_data + 'modeling_dataset_' + SEED +'.csv', index = False)


training_data, test_set = train_test_split( data, test_size = 0.2,
                                            random_state = SEED)

training_set, validation_set = train_test_split( training_data, test_size = 0.1,
                                                 random_state = SEED)

training_set.to_csv( dir_dest + 'training_set.csv', index = False)
validation_set.to_csv( dir_dest + 'validation_set.csv', index = False)
test_set.to_csv( dir_data + 'test_set.csv', index = False)



