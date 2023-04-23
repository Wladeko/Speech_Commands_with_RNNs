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
      origin="https://storage.googleapis.com/kagglesdsdata/competitions/7634/46676/train.7z?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1682509177&Signature=jTadfpVkkZ2N5%2F8ExD7l5lB3sMxRPoteX6WbArpcnr%2BLyqb%2BW9ls4TvgszlpwgKS3KFVe3SuO6EC7jv5W%2FTPB935Vj6FRVZbg97NJlJwEusK2eggn%2FdVZNYEwEVO%2BcfoSZB4HQ3wGuo3oHO5OVVpWSCwgdxOZ%2B8fDjHf9r0oUUvc3EoF4B4pyw7b9jCpjxUG4UoKFNOoWH1ckXwilqK9lMS5paJYfF1Wja0HL2tcRPiHW2m9ooGrzGEL5pYdMYjucuohLTnt%2BJ5qV71CETkbamDsQsZ4GSqlUsK4yeGu1iY2CsJ8AswA%2FL8Zi2qHyi480%2BBzJn1XQJNslKeSA1MDmQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.7z",
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
