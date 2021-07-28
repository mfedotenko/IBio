import BioInterface as bi
import BioDataset as ds

bioDataset = ds.BioDataset(datasetPath="../dataset_1/")
personStructs = bioDataset.getPersonsWithImagePaths()

bioInterface = bi.BioInterface()
#bioInterface.enrolls(personStructs)
#bioInterface.save()
bioInterface.load()
print("1")