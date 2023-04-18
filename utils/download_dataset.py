import os
import pathlib
import shutil
import tensorflow as tf

# Downloading data
data_path = 'data/train/audio'
data_dir = pathlib.Path(data_path)
if not data_dir.exists():
  tf.keras.utils.get_file(
      origin="https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/7634/46676/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1682110210&Signature=Hwt%2B9QK%2FUyOlKBP3dmVKYYdoBuUUThp1KL%2F447Z%2B6vvmj0yoy2keUN7foIQIDADUp%2B4jckXpQILZleqdMPMz8aUdEGCqqnoX%2FAtLvZUO9kPE%2F%2BA5f54R86lVvFwAe%2Bcceu19SCVjeNMuLjUJbgD9ADZNqT7%2B60Z6qyaU%2Bd72u3ufOrAv%2F5OVlgGSelTw5fZRZvn2XQgALULKMT4pBgEvZa6aSpCNxpqwSuzZdGfhh90mbUbdvZ5wkRtuuS0njjs5Z2cGE6ckYI6NRNpkU%2F%2FXHbi5x6%2BxBcFrjvjMs6jYqC%2F2%2BXTK%2BfPU7GPRM82TBsefGWfQ1sycsdo83QnvQSlOZA%3D%3D&response-content-disposition=attachment%3B+filename%3Dtensorflow-speech-recognition-challenge.zip",
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
