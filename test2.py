import actions
import dataset_utils

test_set_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/PredictSet'
model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained-validateationset1000/best_trained_model/my_model'
labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained-validateationset1000/classes.txt'
#model_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained/best_trained_model/my_model'
#labels_path = '/home/osboxes/Desktop/ML/Final_Project/Models/model1-trained/classes.txt'

actions.stream_predict_images(images_path=test_set_path, model_path=model_path, labels_path=labels_path)

dst_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset/'
dataset_utils.expand_dataset(dst_path,
                                 scale_factor=4)
