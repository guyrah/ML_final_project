import dataset_utils as d_utils
import train_model
import actions

path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/Testset'
raw_data_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/FinalDataset/'
prepared_data_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Fullset_prepared'
prepared_data_path2 = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Fullset_prepared2'
test_set_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/Testset'
model_path = '/home/osboxes/PycharmProjects/ML_final_project/Models/inception_retrained_best.pb'
labels_path = '/home/osboxes/PycharmProjects/ML_final_project/Models/inception_retrained_best.txt'

'''
d_utils.prepare_dataset(path,
                        scale_image_size = (28, 28),
                        normalize_values = False,
                        change_to_gray = False,
                        smooth_factor = 1,
                        in_memory = False,
                        target_path = '/home/osboxes/PycharmProjects/ML_final_project/workspace/tmp_images/')
'''

#actions.test_model1(test_set_path, labels_path, model_path, remove_tmp_if_exist=True )

'''
raw_data_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/FinalDataset/'

prepared_data_path2 = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Fullset_prepared'
image_size = (56,56)
change_to_gray = True
actions.create_dataset_from_raw(raw_data_path, prepared_data_path2, scale_image_size=image_size, change_to_gray=change_to_gray)
train_model.train(prepared_data_path2,
                  image_size=image_size,
                  #src_model_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/model_trained_best/my_model',
                  #src_labels_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/classes.txt',
                  summaries_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/summaries')
'''

raw_data_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/FinalDataset/'
prepared_data_path2 = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Fullset_prepared2'
prepared_data_path2  = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/Testset'
image_size = (28,28)
change_to_gray = True
#actions.create_dataset_from_raw(raw_data_path, prepared_data_path2, scale_image_size=image_size, change_to_gray=change_to_gray)
train_model.train(prepared_data_path2,
                  image_size=image_size,
                  #src_model_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/model_trained_best/my_model',
                  #src_labels_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/classes.txt',
                  summaries_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/summaries')


'''




test_set_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/Testset'
test_set_path  = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset_sets/current_work_file0.Dataset'
test_set_path  = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset3'

model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained-validateationset1000/best_trained_model/my_model'
labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained-validateationset1000/classes.txt'
actions.test_model2(test_set_path, labels_path, model_path, remove_tmp_if_exist=True)
'''
'''
test_set_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/PredictSet'
model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1/best_trained_mode/my_model'
labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1/classes.txt'
actions.batch_predict_images(test_set_path, labels_path, model_path, remove_tmp_if_exist=True)
'''


#train_model.retrain_inception()

#actions.create_dataset_from_raw(raw_data_path, prepared_data_path)
#actions.train_inception()

