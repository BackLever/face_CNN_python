from PIL import Image
import os
import pickle
import numpy as np

class_dir = ['\\male', '\\female']
current_dir = os.getcwd()
train_dir = current_dir + '\\face_dataset\\Training'
test_dir = current_dir + '\\face_dataset\\Validation'
    
def load_data(train_num=20000, test_num=5000, save_name='_face.pkl'):
    save_file = current_dir + '\\' + save_name
    if not os.path.exists(save_file):
        dataset = {}
        dataset['X_train'] = []
        dataset['y_train'] = []
        dataset['X_test'] = []
        dataset['y_test'] = []
        for name in class_dir:
            train_dir_ = train_dir + name + '\\'
            test_dir_ = test_dir + name + '\\'
            for i in range(train_num//2):
                img = Image.open(train_dir_ + os.listdir(train_dir_)[i]).convert('L')
                img_resize = img.resize((50, 50))
                pix = np.array(img_resize) / 255
                dataset['X_train'].append(pix)
                if name == '\\male':
                    dataset['y_train'].append([1, 0])
                elif name == '\\female':
                    dataset['y_train'].append([0, 1])

            for i in range(test_num//2):
                img = Image.open(test_dir_ + os.listdir(test_dir_)[i]).convert('L')
                img_resize = img.resize((50, 50))
                pix = np.array(img_resize) / 255
                dataset['X_test'].append(pix)
                if name == '\\male':
                    dataset['y_test'].append([1, 0])
                elif name == '\\female':
                    dataset['y_test'].append([0, 1])

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        dataset['X_train'] = np.expand_dims(dataset['X_train'], axis=1)
        dataset['X_test'] = np.expand_dims(dataset['X_test'], axis=1)
        
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
    
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    return (dataset['X_train'], dataset['y_train']), (dataset['X_test'], dataset['y_test'])