import tensorflow as tf
import json
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
import os, random


class DataHandler:
    def __init__(self, train_json_path, test_json_path, data_path):
        realPath = os.path.realpath(__file__)
        subStr = realPath[:realPath[: realPath[: realPath.rindex('/')].rindex('/')].rindex('/')]
        self.subStr = subStr
        self.train_json_path = train_json_path
        self.test_json_path = test_json_path
        self.data_path = data_path

    def readJson(self, jsonFilePath):
        with open(jsonFilePath, 'r') as rf:
            loadDict = json.load(rf)
            #print(loadDict)
            return loadDict

    def writeJson(self, jsonFilePath, newDict):
        with open(jsonFilePath, 'w') as wf:
            json.dump(newDict, wf)
            print("dump json is over!")

    def prepareImage(self, img, augment=True):
        if augment:
            self.agument(img)
        return img

    def agument(self, img):
        ran = random.random()
        randint = random.randint(0, 2)
        randBri = random.randint(0, 2) / 10
        randCentralCrop = random.randint(6, 8) / 10
        randHue = random.randint(0, 3) / 10  # [0, 0.5]
        # if ran < 0.2:
        #     img = tf.image.flip_left_right(img)
        # ran = random.random()
        # if ran < 0.2:
        #     img = tf.image.flip_up_down(img)
        # ran = random.random()
        if ran < 0.1:
            img = tf.image.adjust_saturation(img, randint)
        ran = random.random()
        if ran < 0.1:
            img = tf.image.adjust_brightness(img, randBri)
        ran = random.random()
        # if ran < 0.2:
        #     img = tf.image.rot90(img)
        #     img = tf.image.rot90(img)
        #     img = tf.image.rot90(img)
        # ran = random.random()
        if ran < 1.0:
            img = tf.image.central_crop(img, central_fraction=randCentralCrop)
        ran = random.random()
        if ran < 0.1:
            img = tf.image.random_hue(img, randHue)
        ran = random.random()
        if ran < 0.1:
            img = tf.image.random_contrast(img, lower=0.5, upper=1.7)
        return img

    def dataGenerator(self, imgLines, batchSize, numClasses):
        n = len(imgLines)
        i = 0
        while True:
            batchImgs = []
            batchLabels = []
            for _ in range(batchSize):
                framesImg5 = []
                if i == 0:
                    np.random.shuffle(imgLines)
                id = imgLines[i]['id']
                keyFrame = imgLines[i]['key_frame']
                keyIndex = 0
                status = imgLines[i]['status']
                # status = tf.one_hot(status, numClasses)
                status = to_categorical(status, num_classes=numClasses)
                # status = tf.expand_dims(status, 0)
                frames = imgLines[i]['frames']
                imgPath = self.data_path + id + '/'
                for index, frame in enumerate(frames):
                    imgName = frame['frame_name']
                    imgData = cv2.imread(imgPath + imgName)
                    # imgData = self.prepareImage(imgData)
                    imgData = np.array(imgData) / 255.
                    imgData = cv2.resize(imgData, (224, 224))
                    if keyFrame == imgName:
                        keyIndex = index
                    framesImg5.append(imgData)
                for _ in range(5 - len(framesImg5)):
                    framesImg5.append(framesImg5[keyIndex])
                i = (i + 1) % n
                batchImgs.append(framesImg5)
                batchLabels.append(status)
            yield (tf.convert_to_tensor(batchImgs), tf.convert_to_tensor(batchLabels))

    def dataPredict(self, imgLines, batchSize):
        n = len(imgLines)
        i = 0
        while True:
            batchImgs = []
            for _ in range(batchSize):
                framesImg5 = []
                id = imgLines[i]['id']
                keyFrame = imgLines[i]['key_frame']
                keyIndex = 0
                frames = imgLines[i]['frames']
                imgPath = self.data_path + id + '/'
                for index, frame in enumerate(frames):
                    imgName = frame['frame_name']
                    imgData = cv2.imread(imgPath + imgName)
                    imgData = tf.image.central_crop(imgData, central_fraction=0.7)
                    imgData = np.array(imgData) / 255.
                    imgData = cv2.resize(imgData, (224, 224))
                    if keyFrame == imgName:
                        keyIndex = index
                    framesImg5.append(imgData)
                for _ in range(5 - len(framesImg5)):
                    framesImg5.append(framesImg5[keyIndex])
                i = (i + 1) % n
                batchImgs.append(framesImg5)
            yield tf.convert_to_tensor(batchImgs)

    def dataAugment(self):
        readJson = self.readJson(self.train_json_path)
        items = readJson['annotations']
        label0Items = []
        label1Items = []
        label2Items = []
        for index, item in enumerate(items):
            status = int(item['status'])
            if status == 0:
                label0Items.append(item)
            elif status == 1:
                label1Items.append(item)
            elif status == 2:
                label2Items.append(item)
        label0 = len(label0Items)
        label1 = len(label1Items)
        label2 = len(label2Items)
        print("label0:", label0, "label1:", label1, "label2:", label2)
        total = label0 + label1 + label2
        print(total)
        print(label0 / total, label1 / total, label2 / total)
        npLabel = np.array([label0, label1, label2])
        maxLabel = npLabel.argmax()
        maxNum = npLabel.max()
        print("maxLabel:", maxLabel, "maxNum:", maxNum)
        for i in range(3):
            if i == 0:
                label = label0
                labelItems = label0Items
            elif i == 1:
                label = label1
                labelItems = label1Items
            else:
                label = label2
                labelItems = label2Items
            num = maxNum - label
            # num = 500
            if num == 0:
                continue
            sliceRet = []
            for ind in range(num):
                randint = random.randint(0, len(labelItems) - 1)
                sliceRet.append(labelItems[randint])
            newSliceList = []
            for imgs in sliceRet:
                id = imgs['id']
                frames = imgs['frames']
                keyFrame = imgs['key_frame']
                status = imgs['status']
                imgPath = self.data_path + id + '/'
                saveId = str((int(id) + 100000))
                imgSavePath = self.data_path + saveId + '/'
                newSlice = {
                    "id": saveId,
                    "key_frame": keyFrame,
                    "status": status,
                    "frames": []
                }
                while True:
                    if os.path.exists(imgSavePath):
                        getId = int(imgSavePath[:-1][imgSavePath[:-1].rfind('/') + 1:])
                        newId = getId + 100000
                        imgSavePath = self.data_path + str(newId) + '/'
                    else:
                        break
                os.mkdir(imgSavePath)
                for img in frames:
                    imgName = img['frame_name']
                    gpsTime = img['gps_time']
                    imgData = cv2.imread(imgPath + imgName)
                    newImgData = self.agument(imgData)
                    cv2.imwrite(imgSavePath + imgName, np.array(newImgData))
                    newFrame = {
                        "frame_name": imgName,
                        "gps_time": gpsTime
                    }
                    newSlice["frames"].append(newFrame)
                newSliceList.append(newSlice)
            items.extend(newSliceList)
        with open(self.subStr + '/data/amap_traffic_augment.json', 'w') as wf:
            json.dump(readJson, wf)

    def framesCount(self):
        readJson = self.readJson(self.train_json_path)
        items = readJson['annotations']
        count = {}
        for index, item in enumerate(items):
            framesLen = len(item['frames'])
            if count.get(framesLen) != None:
                count[framesLen] += 1
            else:
                count[framesLen] = 1
        print("count:", count)

    def framesStatus(self):
        readJson = self.readJson(self.train_json_path)
        items = readJson['annotations']
        count = {}
        for index, item in enumerate(items):
            framesLen = len(item['frames'])
            status = item['key_frame']
            key = str(status) + '-' + str(framesLen)
            if count.get(key) != None:
                count[key] += 1
            else:
                count[key] = 1
        print("key:", count)

    def dataGeneratorV2(self, imgLines, batchSize, numClasses):
        n = len(imgLines)
        i = 0
        while True:
            batchImgs = []
            batchLabels = []
            for _ in range(batchSize):
                framesImg5 = []
                if i == 0:
                    np.random.shuffle(imgLines)
                id = imgLines[i]['id']
                status = imgLines[i]['status']
                # status = tf.one_hot(status, numClasses)
                status = to_categorical(status, num_classes=numClasses)
                # status = tf.expand_dims(status, 0)
                frames = imgLines[i]['frames']
                imgPath = self.data_path + id + '/'
                for index, frame in enumerate(frames):
                    imgName = frame['frame_name']
                    imgData = cv2.imread(imgPath + imgName)
                    # imgData = self.prepareImage(imgData)
                    imgData = np.array(imgData) / 255.
                    imgData = cv2.resize(imgData, (224, 224))
                    framesImg5.append(imgData)
                for _ in range(5 - len(framesImg5)):
                    framesImg5.append(self.agument(framesImg5[len(frames) - 1]))
                i = (i + 1) % n
                batchImgs.append(framesImg5)
                batchLabels.append(status)
            yield (tf.convert_to_tensor(batchImgs), tf.convert_to_tensor(batchLabels))


if __name__ == '__main__':
    realPath = os.path.realpath(__file__)
    subStr = realPath[:realPath[: realPath[: realPath.rindex('/')].rindex('/')].rindex('/')]
    train_json_path = subStr + '/data/amap_traffic_annotations_train.json'
    # train_json_path = subStr + '/data/amap_traffic_augment.json'
    test_json_path = subStr + '/data/amap_traffic_annotations_test.json'
    data_path = subStr + '/data/amap_traffic_train_0712/'
    data_test_path = subStr + '/data/amap_traffic_test_0712/'
    handler = DataHandler(train_json_path, test_json_path, data_path)
    # readJson = handler.readJson(handler.train_json_path)
    # print(readJson)
    # handler.dataAugment()

    handler.framesCount()
    handler.framesStatus()


