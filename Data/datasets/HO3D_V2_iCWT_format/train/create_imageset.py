import glob
import os

subs = glob.glob('Images/*')
subs.sort(key=str.casefold)

is_train = False # True #

idx = 0
if os.path.exists("ImageSets/imageset_train.txt") and is_train:
    os.remove("ImageSets/imageset_train.txt")
if os.path.exists("ImageSets/imageset_test.txt") and not is_train:
    os.remove("ImageSets/imageset_test.txt")
idx_train = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 35, 36, 37, 38, 45, 46, 47, 49, 50, 51]
idx_test = [5, 10, 15, 20, 25, 30, 44, 48, 52]
for sub in subs:
    idx += 1
    if idx not in idx_train and is_train:
        continue
    if idx not in idx_test and not is_train:
        continue
    if not os.path.isdir(os.path.join(os.getcwd(), sub)):
        continue
    print('Processing subfolder ' + sub)
    images = glob.glob(os.path.join(sub, '*'))
    images.sort(key=str.casefold)
    for i in range(len(images)):
        if is_train:
            f = open("ImageSets/imageset_train.txt", "a")
        if not is_train:
            f = open("ImageSets/imageset_test.txt", "a")
        f.write(images[i].replace('Images/', '').replace('.png','\n'))
        f.close()
