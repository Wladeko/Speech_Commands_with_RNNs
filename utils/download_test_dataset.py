import os
import pathlib
import shutil

import tensorflow as tf
from pyunpack import Archive
from tqdm import tqdm

# Downloading data
data_path = 'data/test/audio'
data_dir = pathlib.Path(data_path)
if not data_dir.exists():
  tf.keras.utils.get_file(
      origin="https://storage.googleapis.com/kagglesdsdata/competitions/7634/46676/test.7z?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1682625672&Signature=OTXJDQCRXih%2Fi6bh4%2F93N%2BFh6%2FiDirOiWHSg7CIv0%2FoKd%2BMlrXbfGK07vLwgVbyCSvGVORuMPMzhtoo5IlyEnkRMZxNXdPjH1pYJO13qNw8D8XwijZE703Q1GmeoErR7pPQHDFhzYqPFwaHEiWfzAQfmYEBJutQJUguIlOEv6GGGoyevxUPqUkrQfvfo%2FHQU%2B1%2BpUtZrmu%2F0CFEpUL2zCArL861xoXogP7VfIltKhXzvl%2FyE%2FlTZyIUR1rXDHQ7qT6XBT9udhOV5XL6TACTux1QdLoqPN8U1RPsN4pltSmBlEkVrE7aqQ396hXPQDdRKQ%2Fp76PhLGj2XKAnY2Ipqwg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest.7z",
      cache_dir='.', cache_subdir='data')
Archive('data/train.7z').extractall("data/test")