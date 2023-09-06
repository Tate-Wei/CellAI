from django.shortcuts import render, redirect
import numpy as np
import os

from .plot_prediction import generate_images
from ami.settings import BASE_DIR
from model.src.dataset import STR2NUM
from model.src.utils.utils import ST_PATH

# Create your views here.
def main_page(request):
    pr_file_path = BASE_DIR / 'model' / 'inputs' / 'a_learning' / 'temp' / 'pr_dataset.npy'
    al_file_path = BASE_DIR / 'model' / 'inputs' / 'a_learning' / 'temp' / 'al_label.npy'
    images_info = generate_images(pr_file_path)

    if request.method == "POST":
        value_list = request.POST.getlist("value", [])
        action = int(request.POST.getlist("inputField", [])[0])

        if action == 1:
            images_info = list(reversed(images_info))
            return render(request, os.path.join('main_page', 'main_page.html'), {'images_info': images_info, 'action': action})
        
        elif action == -1:
            return render(request, os.path.join('main_page', 'main_page.html'), {'images_info': images_info, 'action': action})
        
        elif action == 2 or action == 3:
            if value_list:
                try:
                    al_label = np.load(al_file_path, allow_pickle=True).item()
                except FileNotFoundError:
                    al_label = {}

                for value in value_list:
                    idx = value.split("_")[0]
                    label = STR2NUM[value.split("_")[1]]
                    al_label[int(idx)] = {'idx': int(idx), 'label': label}
                
                np.save(al_file_path, al_label)
            
            if action == 2:
                return render(request, os.path.join('main_page', 'main_page.html'), {'images_info': images_info, 'action': -1})
            else:
                return render(request, os.path.join('home', 'training.html'))
            
    else:
        return render(request, os.path.join('main_page', 'main_page.html'), {'images_info': images_info, 'action': -1})

def about(request):
    return render(request, os.path.join('main_page', 'about.html'))

def settings(request):
    content = np.load(ST_PATH, allow_pickle=True).item()
    return render(request, os.path.join('main_page', 'settings.html'), content)

def save(request):
    if request.method == "POST":
        mode = request.POST.getlist("mode", [])[0]
        seg = request.POST.getlist("seg", [])[0]
        lr = request.POST.getlist("lr", [])[0]
        ep = request.POST.getlist("ep", [])[0]
        pre = request.POST.getlist("pre", [])[0]
        content =  {'mode': mode,
                    'seg': seg,
                    'lr': lr,
                    'ep': ep,
                    'pre': pre}
        np.save(ST_PATH, content)
        return redirect('/main_page/')
    else:
        content = np.load(ST_PATH, allow_pickle=True).item()
        return render(request, os.path.join('main_page', 'settings.html'), content)