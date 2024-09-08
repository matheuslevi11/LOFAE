import argparse
import os
import pandas as pd
import torch
import yaml
import tqdm
import warnings
warnings.filterwarnings('ignore')

from torchvision import utils

from datasets import *
from nets import *
from functions import *
from trainer import *

def preprocess(img_path):
    resize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            ])
    normalize = transforms.Normalize(mean=[0.48501961, 0.45795686, 0.40760392], std=[1,1,1])
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)
    img = resize(img_pil)
    if img.size(0) == 1: # grayscale to RGB
        img = torch.cat((img, img, img), dim = 0)
    img = normalize(img)
    return img

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='path to the config file.')
parser.add_argument('--vgg_model_path', type=str, default='./models/dex_imdb_wiki.caffemodel.pt', help='pretrained age classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--img_path', type=str, default='./test/input/', help='test image path')
parser.add_argument('--out_path', type=str, default='./databases/ageKinFace/', help='test output path')
parser.add_argument('--target_age', type=int, default=55, help='Age transform target, interger value between 20 and 70')
opts = parser.parse_known_args()[0]

log_dir = os.path.join(opts.log_path, opts.config) + '/'
if not os.path.exists(opts.out_path):
    os.makedirs(opts.out_path)

config = yaml.safe_load(open('./configs/' + opts.config + '.yaml', 'r'))
img_size = (config['input_w'], config['input_h'])

# Initialize trainer
trainer = Trainer(config)
device = torch.device('cuda')
trainer.to(device)

# Load pretrained model 
if opts.checkpoint:
    trainer.load_checkpoint(opts.checkpoint)
else:
    trainer.load_checkpoint(log_dir + 'checkpoint')

for database in ['KinFaceW-I', 'KinFaceW-II']:
    for folder in ['father-dau', 'father-son', 'mother-dau', 'mother-son']:
        df = pd.read_csv(f'/mnt/heavy/DeepLearning/Datasets/Kinface filenames/{database}_{folder}_files.csv') # ADAPT
        db_folder = f'/mnt/heavy/DeepLearning/Datasets/{database}/images/{folder}/' # ADAPT

        for age in [20, 30, 40, 50, 60]:
            print(f'Generating age {age}')
            for i, row in tqdm.tqdm(df.iterrows()):
                target_path = db_folder + row['filename']
                target_image = row['filename'].split('/')[-1]
                image_name = target_image.split('.')[0] + '_age_' + str(age) + '.jpg'
                
                if os.path.exists(opts.out_path + image_name):
                    continue
                with torch.no_grad():
                    img = preprocess(target_path).unsqueeze(0).to(device)
                    age_modif = torch.tensor(age).unsqueeze(0).to(device)
                    img_modif = trainer.test_eval(img, age_modif, target_age=age, hist_trans=True)  
                    utils.save_image(clip_img(img_modif), opts.out_path + '/' + database + '/images/' + folder + '/' + image_name)