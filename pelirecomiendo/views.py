from calendar import c
from django.shortcuts import render, redirect, get_object_or_404

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login

from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.utils.http import urlsafe_base64_decode
import os

# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval

# 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
# import warnings; warnings.simplefilter('ignore')

def home(request):
    urlExcel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urlExcel = os.path.join(urlExcel, 'dataset')
    urlExcel = os.path.join(urlExcel, 'final_movie_dataset.xlsx')
    # urlExcel = os.path.dirname(os.path.abspath(__file__))
    return HttpResponse(urlExcel)
    return render(request, 'chatbot/index.html')
    # return HttpResponse("vacacuones")