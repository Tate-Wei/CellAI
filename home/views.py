from django.shortcuts import render, redirect
import os

from ami.settings import BASE_DIR
from model.main import main
from .utils import delect_temp_file
#from model.src.utils.utils import Usage

# Create your views here.
FAILED_UPLOAD = False

def home(request):
    global FAILED_UPLOAD
    return render(request, os.path.join('home', 'index.html'), {'failed_upload': FAILED_UPLOAD})

def upload_file(request):
    delect_temp_file('prediction')

    global FAILED_UPLOAD
    if request.method == "POST":
        target_file =request.FILES.get("file", None)
        if not target_file:
            FAILED_UPLOAD = True
            return render(request, os.path.join('home', 'index.html'), {'failed_upload': FAILED_UPLOAD})
        
        destination = open(BASE_DIR / 'model' / 'inputs' / 'prediction' / target_file.name, 'wb+')
        for chunk in target_file.chunks():
            destination.write(chunk)  
        destination.close()
        FAILED_UPLOAD = False
        return render(request, os.path.join('home', 'training.html'))
    
def training(request):
    global FAILED_UPLOAD
    if os.path.exists(BASE_DIR / 'model' / 'inputs' / 'a_learning' / "temp" / "al_label.npy"):
        delect_temp_file('plots')
        main(False, True, False, BASE_DIR / 'model' / 'outputs' / 'checkpoints')
        try:
            print("+++ Active Learning Finished +++")
        except:
            FAILED_UPLOAD = True
            delect_temp_file('temp')
            return render(request, os.path.join('main_page', 'main_page.html'), {'failed_upload': FAILED_UPLOAD})
        else:
            delect_temp_file('temp')
            FAILED_UPLOAD = False
            return render(request, os.path.join('home', 'prediction.html'))
        
    elif os.path.exists(BASE_DIR / 'model' / 'inputs' / 'training' / 'prediction.seg'):
        delect_temp_file('plots')
        delect_temp_file('temp')

        print("+++ Pre-Training Started +++")
        main(True, False, False, None)
        print("+++ Pre-Training Finished +++")

        return render(request, os.path.join('home', 'prediction.html'))
    
    else:
        print("+++ No Training +++")
        delect_temp_file('plots/conf')
        delect_temp_file('temp')
        return render(request, os.path.join('home', 'prediction.html'))

def prediction(request):
    global FAILED_UPLOAD
    main(False, False, True, BASE_DIR / 'model' / 'outputs' / 'checkpoints')
    try:
        print("+++ Prediction Finished +++")
    except:
        FAILED_UPLOAD = True
        return render(request, os.path.join('main_page', 'main_page.html'), {'failed_upload': FAILED_UPLOAD})
    else:
        FAILED_UPLOAD = False
        return redirect('/main_page/')
