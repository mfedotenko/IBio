import FaceEngine as fe
from tqdm import tqdm
import numpy

lunasdk_path = "../lunasdk/"
data_path = lunasdk_path + "data/"
license_conf_filename = data_path + "license.conf"
faceengine_conf_filename = data_path + "faceengine.conf"

class Luna():
    def __init__(self, data_path=data_path, license_conf_path=license_conf_filename, faceengine_conf_filename=faceengine_conf_filename):
        self.faceEngine = fe.createFaceEngine(data_path)
        # activation
        license = self.faceEngine.getLicense()
        activationCode = self.faceEngine.activateLicense(license, license_conf_path)
        if not activationCode:
            print("failed to activate license!")
            exit(-1)
        config = fe.createSettingsProvider(faceengine_conf_filename)
        config_path = config.getDefaultPath()
        print("Config settings: DefaultPath {0}".format(config_path))
        config.setValue("system", "verboseLogging", 1)
        self.faceEngine.setSettingsProvider(config)
        val = config.getValue("system", "verboseLogging")
        print("Config settings: \"system\", \"verboseLogging\" = {0}".format(val))
        #
        self.detector = self.faceEngine.createDetector(fe.FACE_DET_V3)
        self.warper = self.faceEngine.createWarper()
        self.extractor = self.faceEngine.createExtractor()

    def imageConvert(self, imagePath):
        image = fe.Image()
        image.load(imagePath)
        err, face = self.detector.detectOne(image, image.getRect(), fe.DetectionType(fe.dt5Landmarks))
        if not face.landmarks5_opt.isValid():
            print('failed to do detection')
            return None
        det, landmarks5 = face.detection, face.landmarks5_opt.value()
        transformation = self.warper.createTransformation(det, landmarks5)
        warpResult = self.warper.warp(image, transformation)
        if warpResult[0].isError:
            print("Failed image warping.")
            return None
        warpImage = warpResult[1]
        return warpImage

    def extract(self, imagePath):
        image = self.imageConvert(imagePath)
        if image is None: return None
        descriptor = self.faceEngine.createDescriptor()
        res, _ = self.extractor.extractFromWarpedImage(image, descriptor)
        if res.isError:
            print("Failed to extract descriptor1, reason: ", res.what)
            return None
        err, vector = descriptor.save()
        return vector

    def match(self, descriptor1, descriptor2):
        matcher = self.faceEngine.createMatcher()
        desc1 = self.faceEngine.createDescriptor()
        desc2 = self.faceEngine.createDescriptor()
        desc1.load(descriptor1)
        desc2.load(descriptor2)
        err, value = matcher.match(descriptor1, descriptor2)
        return {"distance": value.distance, "similarity": value.similarity}

    def runExtract(self, images):
        extractor = self.faceEngine.createExtractor()
        convertImages = []
        for i in tqdm(range(len(images)), desc="runExtract"):
            image = images[i]
            im = fe.Image()
            im.load(image)
            if not im.isValid():
                print(im, ' was not loaded')
                exit(1)
            convertImage = self.imageConvert(im)
            if convertImage is None:
                print('failed to detect and warp', image)
                continue
            descriptor = self.faceEngine.createDescriptor()
            res, _ = extractor.extractFromWarpedImage(convertImage, descriptor)
            if res.isError:
                print("Failed to extract descriptor1, reason: ", res.what)
                exit(-1)
            err, vector = descriptor.save()
            convertImages.append(vector)
        return numpy.asarray(convertImages)

    def runMatch(self, probeDescriptors, galleryDescriptors):
        matcher = self.faceEngine.createMatcher()
        result = []
        for i in tqdm(range(len(probeDescriptors)), desc="runMatch"):
            t = []
            for j in range(len(galleryDescriptors)):
                descriptor1 = self.faceEngine.createDescriptor()
                descriptor2 = self.faceEngine.createDescriptor()
                descriptor1.load(probeDescriptors[i], len(probeDescriptors[i]))
                descriptor2.load(galleryDescriptors[j], len(galleryDescriptors[j]))
                err, value = matcher.match(descriptor1, descriptor2)
                t.append({"distance": value.distance, "similarity": value.similarity})
            result.append(t)
        return numpy.asarray(result)

    def run_extractor(self, image_paths):
        extractor = self.faceEngine.createExtractor()
        im_descriptors = []
        im_ids = []
        face_ids = []
        im_id = 0
        for i in tqdm(range(len(image_paths)), desc="run_extractor"):
            image = image_paths[i]
            im = fe.Image()
            im.load(image)
            if not im.isValid():
                print(im, ' was not loaded')
                exit(1)

            wimg = self.detect_and_warp(im)
            if wimg is None:
                print('failed to detect and warp', image)
                continue
            descriptor1 = self.faceEngine.createDescriptor()
            res1, _ = extractor.extractFromWarpedImage(wimg, descriptor1)
            if res1.isError:
                print("Failed to extract descriptor1, reason: ", res1.what)
                exit(-1)

            err, vec_saved = descriptor1.save()
            im_descriptors.append(vec_saved)

            im_ids.append(im_id)
            face_ids.append(image.split('/')[-2])

            im_id += 1

        return numpy.asarray(im_ids), numpy.asarray(face_ids), numpy.asarray(im_descriptors)

    def run_matcher(self, probe_descriptors, gallery_descriptors):
        matcher = self.faceEngine.createMatcher()
        result_distance = []
        result_similarity = []
        for i in tqdm(range(len(probe_descriptors)), desc="run_matcher"):
            t1 = []
            t2 = []
            for j in range(len(gallery_descriptors)):
                descriptor1 = self.faceEngine.createDescriptor()
                descriptor2 = self.faceEngine.createDescriptor()
                descriptor1.load(probe_descriptors[i], len(probe_descriptors[i]))
                descriptor2.load(gallery_descriptors[j], len(gallery_descriptors[j]))

                err1, value1 = matcher.match(descriptor1, descriptor2)
                t1.append(value1.distance)
                t2.append(value1.similarity)

            result_distance.append(t1)
            result_similarity.append(t2)

        return numpy.asarray(result_distance), numpy.asarray(result_similarity)

