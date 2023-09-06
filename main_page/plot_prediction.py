import numpy as np
import matplotlib.pyplot as plt
import os

from ami.settings import BASE_DIR
from model.src.dataset import NUM2STR
from model.src.utils.utils import load_settings

IMG_DIR = BASE_DIR / 'static'

def generate_images(file_path, figsize=(5, 5)):
    pred_data = np.load(file_path, allow_pickle=True).item()
    
    images_info = []
    for i in pred_data.keys():
        ph_path = os.path.join('plots', f'ph_{i}.png')
        amp_path = os.path.join('plots', f'amp_{i}.png')
        cl_ph_path = os.path.join('plots', f'cl_ph_{i}.png')
        #hm_path = os.path.join('plots', f'hm_{i}.png')
        hm_ori_path = os.path.join('plots', f'hm_ori_{i}.png')
        
        if not os.path.exists(IMG_DIR / ph_path):
            mode = load_settings()['mode']
            ori_ph = pred_data[i]['ori_ph']
            ori_amp = pred_data[i]['ori_amp']
            cl_ph = pred_data[i][mode]
            #heatmap = pred_data[i]['heatmap']
            heatmap_ori = pred_data[i]['heatmap+ori_img']
            
            plt.switch_backend('AGG')
            plt.figure(1, figsize=figsize)
            plt.imshow(ori_ph)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(IMG_DIR / ph_path)

            plt.figure(2, figsize=figsize)
            plt.imshow(ori_amp)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(IMG_DIR / amp_path)
            
            plt.figure(3, figsize=figsize)
            plt.imshow(cl_ph)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(IMG_DIR / cl_ph_path)

            plt.figure(4, figsize=figsize)
            plt.imshow(heatmap_ori)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(IMG_DIR / hm_ori_path)
            plt.close('all')
        
        images_info.append({'index': i,
                            'ori_ph': ph_path,
                            'ori_amp': amp_path,
                            'cl_ph': cl_ph_path,
                            'hm_ori': hm_ori_path,
                            'label': NUM2STR[pred_data[i]['label']],
                            'uncertainty': round(pred_data[i]['conf'], 4)})
        
    return images_info