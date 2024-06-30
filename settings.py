import os, json
import boto3
import torch

boto_session = boto3.Session(region_name='ap-southeast-1')
s3 = boto_session.client('s3')

BUCKET_NAME = os.environ.get('BUCKET_NAME')
METADATA_KEY = 'data/covid-csv-metadata.json'
metadata = json.loads(s3.get_object(Bucket=BUCKET_NAME, Key=METADATA_KEY)['Body'].read().decode('utf-8'))
input_schema = metadata['schema']['input_features']

PRETRAINED_MODEL_NAME = 'bert-base-uncased'
RANK = int(os.environ.get('RANK', -1))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
DEVICE = f'cuda:{LOCAL_RANK}' if torch.cuda.is_available() else 'cpu'

TRAIN_DATA_S3_URL = f's3://{BUCKET_NAME}/data/covid-csv/training'
TEST_DATA_S3_URL = f's3://{BUCKET_NAME}/data/covid-csv/testing'
N_SAMPLES = metadata['dataset_size']
TRAIN_FILES = N_SAMPLES * 4 // 5 // 16 + 1
TEST_FILES = N_SAMPLES // 5 // 16 + 1
BATCH_SIZE = metadata['batch_size']
TEST_DATASET_SIZE = metadata['test_size']
TRAIN_DATASET_SIZE = metadata['train_size']
MODEL_OUTPUT_S3_URL = 's3://{BUCKET_NAME}/test/model-output'
EPOCHS = 1
LR = 0.001
