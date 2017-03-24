import dataset_utils
import train_model
import shutil
import run_model
import os

def create_dataset_from_raw(src_path, dst_path, scale_image_size=(-1,-1), normalize_values=False, change_to_gray=False, smooth_factor=1, in_memory=False):
    '''
    #scale_image_size = (28, 28)
    normalize_values = False
    change_to_gray = True
    smooth_factor = 1
    in_memory = False
    '''
    shutil.rmtree(dst_path)
    dataset_utils.prepare_dataset(src_path,
                                  scale_image_size=scale_image_size,
                                  normalize_values=normalize_values,
                                  change_to_gray=change_to_gray,
                                  smooth_factor=smooth_factor,
                                  in_memory=False,
                                  target_path=dst_path)
    dataset_utils.expand_dataset(dst_path,
                                 scale_factor=2,
                                 add_square_page_background_factor=3,
                                 square_page_path='/home/osboxes/PycharmProjects/ML_final_project/DataBase/square_paper3.jpg')

def train_inception():
    train_model.retrain_inception()


def test_model(images_path, labels_path, model_path, remove_tmp_if_exist=True):
    ''''
            Test model where the graph was exported in pb format
    '''
    workspace_path = '/home/osboxes/Desktop/ML/Final_Project/Workspace/test_model_data'
    output_path = '/home/osboxes/PycharmProjects/ML_final_project/workspace'
    if os.path.exists(workspace_path) and remove_tmp_if_exist:
        shutil.rmtree(workspace_path)

    dataset_utils.prepare_dataset(images_path,
                    scale_image_size=(28, 28),
                    normalize_values=False,
                    change_to_gray=False,
                    smooth_factor=1,
                    in_memory=False,
                    target_path=workspace_path)

    run_model.test_model(workspace_path, labels_path, model_path, output_path)



def test_model2(images_path, labels_path, model_path, remove_tmp_if_exist=True):
    ''''
        Test model where the graph was saved as checkpoints
    '''
    workspace_path = '/home/osboxes/Desktop/ML/Final_Project/Workspace/test_model_data'
    output_path = '/home/osboxes/PycharmProjects/ML_final_project/workspace2'
    if os.path.exists(workspace_path) and remove_tmp_if_exist:
        shutil.rmtree(workspace_path)

    dataset_utils.prepare_dataset(images_path,
                    scale_image_size=(28, 28),
                    normalize_values=False,
                    change_to_gray=True,
                    smooth_factor=1,
                    in_memory=False,
                    target_path=workspace_path)

    run_model.test_model2(workspace_path, labels_path, model_path, output_path)

def batch_predict_images(images_path, labels_path, model_path, remove_tmp_if_exist=True):
    workspace_path = '/home/osboxes/Desktop/ML/Final_Project/Workspace/test_model_data'
    if os.path.exists(workspace_path) and remove_tmp_if_exist:
        shutil.rmtree(workspace_path)

    dataset_utils.prepare_dataset(images_path,
                    scale_image_size=(28, 28),
                    normalize_values=False,
                    change_to_gray=True,
                    smooth_factor=1,
                    in_memory=False,
                    target_path=workspace_path)

    run_model.predict2(workspace_path, labels_path, model_path, image_size=(28,28))


def stream_predict_images(images_path, labels_path, model_path):
    scale_image_size = (28,28)
    normalize_values = False
    change_to_gray = True
    smooth_factor = 1
    run_model.predict_unprepared_data2(imagesPath=images_path,
                                       labelsFullPath=labels_path,
                                       modelFullPath=model_path,
                                       scale_image_size=scale_image_size,
                                       normalize_values=normalize_values,
                                       change_to_gray=change_to_gray,
                                       smooth_factor=smooth_factor)
