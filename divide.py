import os, random, shutil


def make_dir(source, target):
    dir_names = os.listdir(source)
    for names in dir_names:
        for i in ['train', 'valid']:
            path = target + '/' + i + '/' + names
            if not os.path.exists(path):
                os.makedirs(path)


def divideTrainValiTest(source, target):
    pic_name = os.listdir(source)

    s = list(range(1053))

    random.seed(1313)
    random.shuffle(s)
    
    for classes in pic_name:

        pic_classes_name = sorted(os.listdir(os.path.join(source, classes)))

        # print(pic_classes_name[0:10])
        # print(len(pic_classes_name))
        
        # 8：1：1 
        index = s[0:int(0.9 * len(s))]
        print(index[0:10])
        train_list = [pic_classes_name[i] for i in index]
        index = s[int(0.9 * len(s)):]
        print(index[0:10])
        valid_list = [pic_classes_name[i] for i in index]
        # index = s[int(0.9 * len(s)):]
        # print(index[0:10])
        # test_list = [pic_classes_name[i] for i in index]

        print(len(train_list))
        print(len(valid_list))
        # print(len(test_list))
        
        for train_pic in train_list:
            shutil.copyfile(source + '/' + classes + '/' + train_pic, target + '/train/' + classes + '/' + train_pic)
        for validation_pic in valid_list:
            shutil.copyfile(source + '/' + classes + '/' + validation_pic,
                            target + '/valid/' + classes + '/' + validation_pic)

if __name__ == '__main__':
    filepath = r'./SEG_Train_Datasets'
    dist = r'./Images_Train90Valid10'
    make_dir(filepath, dist)
    divideTrainValiTest(filepath, dist)
