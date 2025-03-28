# Place all dataset constant variables here.
from pathlib import Path


# COCO 2014
MSCOCO14_TRAIN_IMG_PATH = Path('[your path]/COCO2014/train2014/')
MSCOCO14_VAL_IMG_PATH = Path('[your path]/COCO2014/val2014/')
MSCOCO14_TRAIN_METADATA = Path('./src/data/mscoco/train_anno.json')
MSCOCO14_VAL_METADATA = Path('./src/data/mscoco/val_anno.json')



# IU X-ray
IUXRAY_TRAIN_IMG_PATH = Path('[your path]/IU-XRay/images')
IUXRAY_TEST_IMG_PATH = Path('[your path]/IU-XRay/images')
IUXRAY_TRAIN_METADATA = Path('./src/data/iuxray/training_set.csv')
IUXRAY_TEST_METADATA = Path('./src/data/iuxray/testing_set.csv')




