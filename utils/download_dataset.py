import os
import pathlib
import shutil
import tensorflow as tf

from pyunpack import Archive

# Downloading data
data_path = 'data/train/audio'
data_dir = pathlib.Path(data_path)
if not data_dir.exists():
  tf.keras.utils.get_file(
      origin="https://storage.googleapis.com/kagglesdsdata/competitions/7634/46676/train.7z?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1682110428&Signature=cq%2FPQDH8watXYiVGtl2xPd0cRyHkS3F4QyPYFhn6GtyZ85MJueaOUnWhcn1lzww%2BCkwbUvDXhrpqJ4EYmh%2F6tsicinc%2FC08ZyvZ1SkPDC26yQvSGsLvjuX5PVeNT2ucFL1WaFUlXsyHX9KbqgsOeNgqO5nzsLAyJLTyLDrTKsQmxq4v4r8ryAgrXB4NZCQR%2BYeZJEU8A2oWvmkHo9vtaa7xPtSmqs%2Flm2bKvQq3Fjcl8r8%2BTFJwjBDwgfSFKzpD8QJ4OXOh2x%2FEDVptrJAbQ5LiH5K8JydEnWBtyBLUHxTJg%2Bl5pH%2Flku1votcwMMFpHFPZl3u5PopYfq%2Fye6%2FxfwA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.7z",
      cache_dir='.', cache_subdir='data')
Archive('data/train.7z').extractall("data")
  
# Moving validation files
val_path = 'data/val/audio'
val_dir = pathlib.Path(val_path)
isExist = os.path.exists(val_path)
if not isExist:
  os.makedirs(val_path) 
  val_file = open('data/train/validation_list.txt', 'r')
  val_count = 0
  for line in val_file:
    src_path = "data/train/audio/"+line.strip()
    dst_path = "data/val/audio/"+line.strip()
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)
      shutil.move(src_path, dst_path)
      val_count += 1
  val_file.close()
  print("Moved {} files to the validation directory".format(val_count))

# Moving test files
test_path = 'data/test/audio' 
test_dir = pathlib.Path(test_path)
isExist = os.path.exists(test_path)
if not isExist:
  os.makedirs(test_path) 
  test_file = open('data/train/testing_list.txt', 'r')
  test_count = 0
  for line in test_file:
    src_path = "data/train/audio/"+line.strip()
    dst_path = "data/test/audio/"+line.strip()
    if not os.path.exists(dst_path):
      os.makedirs(dst_path)
      shutil.move(src_path, dst_path)
      test_count += 1
  test_file.close()
  print("Moved {} files to the test directory".format(test_count))
