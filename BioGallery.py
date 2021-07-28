import pickle

class bioGallery():

    def __init__(self, mode="All"):
        self.mode = mode
        self.gallery = {}
        self.savePath = "./output/bioGallery.pkl"

    def getMode(self):
        return self.mode

    def getGallery(self):
        return self.gallery

    def add(self, person, descriptor):
        personDescriptors = self.gallery.get(person)
        if personDescriptors is None: personDescriptors = []
        personDescriptors.append(descriptor)
        self.gallery.update({person: personDescriptors})

    def adds(self, personStructs):
        for personStruct in personStructs:
            self.add(personStruct.person, personStruct.descriptor)

    def save(self):
        with open(self.savePath, 'wb') as f:
            pickle.dump(self.gallery, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        with open(self.savePath, 'rb') as f:
            self.gallery = pickle.load(f)
