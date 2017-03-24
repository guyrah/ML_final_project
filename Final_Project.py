import actions
import train_model


MODE = 'TRAIN'
MODE = 'TEST'
MODE = 'PREDICT'
MODE = 'EXPLAIN'

model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model0-trained/best_trained_model/my_model'
labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model0-trained/classes.txt'
#model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained/best_trained_model/my_model'
#labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained/classes.txt'
model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model2-trained/best_trained_model/my_model'
labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model2-trained/classes.txt'

if MODE == 'TRAIN':
    raw_data_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/FinalDataset/'

    prepared_data_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Fullset_prepared'
    image_size = (28, 28)
    change_to_gray = True
    actions.create_dataset_from_raw(raw_data_path, prepared_data_path, scale_image_size=image_size,
                                    change_to_gray=change_to_gray)
    train_model.train(images_path=prepared_data_path,
                      image_size=image_size,
                      # src_model_path=model_path,
                      # src_labels_path=labels_path,
                      summaries_path='/home/osboxes/Desktop/ML/Final_Project/Models/model1/summaries')
elif MODE == 'TEST':
    test_set_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset'
    #test_set_path = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/Testset'
    actions.test_model2(images_path=test_set_path, labels_path=labels_path, model_path=model_path, remove_tmp_if_exist=True)
elif MODE == 'PREDICT':
    images_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Pictures'
    actions.stream_predict_images(images_path=images_path, model_path=model_path, labels_path=labels_path)
    #actions.batch_predict_images(images_path=images_path, labels_path=labels_path, model_path=model_path, remove_tmp_if_exist=True)
elif MODE == 'EXPLAIN':
    image_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Pictures/101.jpg'
    actions.explain_prediction(image_path)