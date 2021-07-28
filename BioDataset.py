import os
import cv2
import numpy
import base64
from tqdm import tqdm

class BioDataset():
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath

    def loadImageFromPath(self, path):
        return cv2.imread(path)

    def loadImageFromUri(self, uri):
        encodeData = uri.split(',')[1]
        nparr = numpy.fromstring(base64.b64decode(encodeData), numpy.uint8)
        image = cv2.imdecode(nparr)
        return image

    def getImagePaths(self):
        imagePaths = []
        for name1 in os.listdir(self.datasetPath):
            path = os.path.join(self.datasetPath, name1)
            for name2 in os.listdir(path):
                d = os.path.join(path, name2)
                for im in os.listdir(d):
                    im = os.path.join(d, im)
                    imagePaths.append(im)
        return imagePaths

    def getPersonsWithImagePaths(self):
        imagePaths = self.getImagePaths()
        resultSamples = []
        imageId = 0
        for i in tqdm(range(len(imagePaths)), desc="getPersonsWithImagePaths"):
            imagePath = imagePaths[i]
            person = imagePath.split('/')[-2]
            resultSamples.append({"id": imageId, "person": person, "imagePath": imagePath})
            imageId += 1
        return resultSamples
