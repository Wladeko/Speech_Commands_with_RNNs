import os
import pathlib
import shutil
import tensorflow as tf

# Downloading data
data_path = 'data/train/audio'
data_dir = pathlib.Path(data_path)
if not data_dir.exists():
  tf.keras.utils.get_file(
      origin="https://storage.googleapis.com/kagglesdsdata/competitions/7634/46676/train.7z?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1681547247&Signature=aFCjRW0p6YvcYjT1WuNILjnV89x8k4ErRIaGqfwJXFinRUXznNH5lEjo313UbgKxtRrDGloswFJqC%2B87xXeNC25kEFAfx9fDTVHvz7tnNmFTYXKP8OA%2FL%2FvD9eSRBR6SPpSEEa79D%2FNeH2Cy6pe1%2BIJ%2Bx6k1QqiCN0OMVIwIk1oGlmTjBTp2EN1XEkrrZ7CzajMplMX7UQavYBJASnLb6cT5jkhh5Ctc6KO%2FJUuOKNKr65XSI309IOB8NrFydwH7dvCJdnnpPczRtzOCGyITb41Q89rTmN8T9OMf%2By45qHnifWkkj91WewvNLWDlZ9ndUQhDAYYkqDyPf7mbH%2BhC5Q%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.7z",
      cache_dir='.', cache_subdir='data')
  
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
