import os
import shutil
from pyunpack import Archive

def merge_dirs(old_path, new_path):
    os.chdir(old_path)

    for i in xrange(51):
        if os.path.exists(str(i)):
            ##print('exist')

            if not os.path.isdir(str(i)):
                raise NameError('dir name exists as file')
        else:
            os.mkdir(str(i))
            #print 'created'

    # validate new path for all dirs and no dir 22

    os.chdir(new_path)
    '''

    for i in xrange(51):
        i = str(i)
        dirs = os.listdir('.')
        if '22' in dirs:
            raise NameError('Dir 22 exist')
        elif (i != '22') and (i not in dirs):
            raise NameError('dir {} does not exist'.format(i))
    '''
    # Copy dirs
    dirs = os.listdir('.')

    if len(dirs) > 0:
        dirs.sort()
    valid_dirs = range(51)
    valid_dirs = map(str, valid_dirs)

    for dir in dirs:
        if os.path.isdir(dir) and dir in valid_dirs:
            os.chdir(dir)
            #print old_path + '/' + dir
            dirMaxFile = int(getMaxFileInDir(old_path + '/' + dir))
            #print 'dir {} has {} files'.format(dir, dirMaxFile)

            for file in os.listdir('.'):
                if os.path.isfile(file):
                    srcPath = new_path + '/' + dir + '/' + file
                    dstPath = old_path + '/' + dir + '/' + str(dirMaxFile + 1) + '.' + getFileSuffix(file).lower()
                    shutil.copy(srcPath, dstPath)
                    #shutil.move(srcPath, dstPath)
                    dirMaxFile += 1


            os.chdir('..')


def getFileSuffix(file):
    return file.split('.')[-1]

def getMaxFileInDir(path):
    #print path
    #print os.listdir(path)
    files = os.listdir(path)

    for i in xrange(len(files)):
        files[i] = int(str(files[i]).split('.')[0])

    if len(files) == 0:
        return 0
    else:
        #print max(files)
        return str(max(files)).split('.')[0]

#
dataset_path = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset_sets'
FINAL_DATASET_PATH = '/home/osboxes/Desktop/ML/Final_Project/Datasets/Testset2'
WORK_DIR_PATH = '/home/osboxes/PycharmProjects/ML_final_project/DataBase/current_work_dir'
#TMP_FILE_NAME = '/home/osboxes/PycharmProjects/ML_FinalProject/DataBase/current_work_file'


for i, file in enumerate(os.listdir(dataset_path)):
    current_path = dataset_path + '/' + str(file)
    file_ending = file.split('.')[-1]
    current_work_file_path = dataset_path + '/current_work_file' + str(i) + '.' + file_ending
    shutil.move(current_path, current_work_file_path)


    if os.path.isdir(current_work_file_path):
        merge_dirs(FINAL_DATASET_PATH, current_work_file_path)
    elif os.path.isfile(current_work_file_path):
        Archive(current_work_file_path).extractall(WORK_DIR_PATH, auto_create_dir=True)

        extracted_list_dir = os.listdir(WORK_DIR_PATH)

        if len(extracted_list_dir) > 1:
            merge_dirs(FINAL_DATASET_PATH, WORK_DIR_PATH)
        else:
            merge_dirs(FINAL_DATASET_PATH, WORK_DIR_PATH+'/'+extracted_list_dir[0])

    shutil.rmtree(WORK_DIR_PATH, ignore_errors=True)

