from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .util import classify_image
from pathlib import Path
import random
from .models import orders

from django.contrib.auth.decorators import login_required
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
# Create your views here.

# def main(request):
#     return render(request,'index.html')


def cals(request):
    return render(request, 'calorieCalc.html')


def exercise(request):
    return render(request, 'exercise.html')


def ingredient(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location='media')
        filename = fs.save(myfile.name, myfile)
        temp = classify_image(str(BASE_DIR)+'\\media\\'+filename)
        context = {
            'data': temp['plant'],
            'pb': temp['probablity'],
            'cals': random.randrange(50, 200),
            'result': True
        }
        return render(request, 'ingredientParser.html', context=context)
    return render(request, 'ingredientParser.html')


def recepies(request):

    return render(request, 'recipes.html')


def addfoods(request):
    return render(request, 'addFood.html')


def home(request):
    return render(request, 'home.html')


def main(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location='media/order_images')
        filename = fs.save(myfile.name, myfile)
        # temp = classify_image(str(BASE_DIR)+'\\media\\'+filename)
        a = orders()
        a.order_image = filename
        a.user_id = request.user
        a.save()
    return render(request, 'findAndOrder.html')
