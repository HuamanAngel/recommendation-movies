from calendar import c
import json
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
import joblib
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
# import warnings; warnings.simplefilter('ignore')

def trainingModel(request):
    urlExcel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urlExcel = os.path.join(urlExcel, 'dataset')
    urlExcel = os.path.join(urlExcel, 'final_movie_dataset.xlsx')    
    md = pd.read_excel(urlExcel)
    print(md.head())
    md.drop(['cast'],axis = 'columns', inplace=True)
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    print(md.head())
    
    # Sistema de recomendacion simple
    #conteo de votos por película
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')    

    #Promedio de votos por película
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    
    #C es el promedio general de votos
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    #se crea un nuevo dataset, solo con las películas cuyo conteo de votos sean mayor a m, y no tengan valores nulos
    qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    print(qualified.shape)
    
    #función que da una calificación ponderada a las películas para saber cuales son las películas con mejor promedio de puntación
    #en base a conteo de votos y promedio de calificación por película
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)    
    
    #se le añade la columna de calificación ponderada al dataset qualified
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    # qualified.head(15)
    
    
    # Recomendador simple con un percentil variable
    s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)        
    
    
    # Recomendador basado en contenido
    #Se llama al dataset de links small para reducir el dataset
    
    urlExcel2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urlExcel2 = os.path.join(urlExcel2, 'dataset')
    urlExcel2 = os.path.join(urlExcel2, 'links_small.csv')    
    
    links_small = pd.read_csv(urlExcel2)
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    md = md.drop([19730, 29503, 35587])        
    #Se convierte la columna ID del dataset a entero
    md['id'] = md['id'].astype('int')    
    
    #Se crea un nuevo dataset, usando los IDs de links small para reducir el dataset a un tamaño considerable
    smd = md[md['id'].isin(links_small)]
    
    #En caso de que haya espacios en blanco en la columna de descripcion en español, se rellenan con fillna
    smd['description'] = smd['overview_es'].fillna('')    
    #Se usa el TF IDF para extraer las palabras de la columna 
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])    
    
    #Se utiliza el lineal kernel, que usa la semejanza de coseno para traer las películas más similares
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    #Se genera una matriz con los títulos y los IDs
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    
    
    
    
    urlExcel3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urlExcel3 = os.path.join(urlExcel3, 'dataset')
    urlExcel3 = os.path.join(urlExcel3, 'credits.csv')    
    
    #se llama al dataset credits que contiene información extra como el reparto de actores, director, entre otra información
    credits = pd.read_csv(urlExcel3)
    #Se conbina con el dataset que ya teniamos
    md = md.merge(credits, on='id')
    smd = md[md['id'].isin(links_small)]
    #Se utiliza el método literal_eval para que convierta o interprete el contenido de las columnas cast y crew 
    #en las estructuras que se asemejan, en este caso, diccionarios.
    smd['cast'] = smd['cast'].apply(literal_eval)
    smd['crew'] = smd['crew'].apply(literal_eval)
    smd['keywords'] = smd['keywords'].apply(literal_eval)
    #Se cuenta el total de elementos de las columnas cast y crew 
    smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
    smd['crew_size'] = smd['crew'].apply(lambda x: len(x))    
    #Esta función sirve para localizar el nombre del director de una película
    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan
    #Con la función anterior se guarda en una nueva columna el director de cada película
    smd['director'] = smd['crew'].apply(get_director)
    
    #En la columna 'cast', se conservan solo los nombres del elenco en una lista
    smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    #Solo se conservan los 3 actores más importantes del elenco
    smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
    
    #Al igual que en la columna cast, se conservan solo los nombres de palabras clave.
    smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])    
                        
    #Para que el algoritmo TF-IDF funcione mejor en esta columna, se quitan los espacios a cada nombre del elenco
    smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    
    #Se quita el espacio a los nombres de los directores y se ponen 3 veces para que tenga más fuerza al ser analizado por el TD-IDF
    smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    smd['director'] = smd['director'].apply(lambda x: [x,x, x])    
                            
    #Se extraen las palabras clave
    s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'keyword'
    
    #Se realiza un recuento de las palabras clave para usarlas en el TF-IDF
    s = s.value_counts()
    #Se descartan las palabras clave que solo aparecen una vez
    s = s[s > 1]
    #Se utiliza la librería SnowballStemmer para que las palabras derivadas regresen a su raiz
    stemmer = SnowballStemmer('english')

    #Se filtran todas las palabras clave
    def filter_keywords(x):
        words = []
        for i in x:
            if i in s:
                words.append(i)
        return words        
                    
    #se usa la función de filter_keywords para poner una lista solo las palabras clave.
    smd['keywords'] = smd['keywords'].apply(filter_keywords)
    #Luego a estas palabras clave, vuelven a su raíz para tener una mayor generalización de palabras
    smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    #Se le quitan los espacios a las palabras clave con  2 o más palabras.
    smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])                                
        
    #Se forma una nueva fila, con toda la información correspondiente: Palabras clave, reparto, director y género
    smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
    smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))    
    
    #Se realiza el algortimo con esta columna y se guarda en la matriz count_matriz
    count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])    
    #Se realiza la similaridad de coseno
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    #Se reestablece la variable smd y los valores titles e indices para la función de recomendación
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])    
        
    # Todas las películas en base a calificaciones
    joblib.dump( qualified, 'model_ia/qualified.pkl' )
    
    # Todas las películas en base a generos
    joblib.dump( [gen_md,indices,cosine_sim,titles,smd], 'model_ia/model_recomendation.pkl' )
    return HttpResponse("Modelo creado correctamente")


def getRecomendations(request):
    urlModel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urlModel = os.path.join(urlModel, 'model_ia')
    print(request.GET['movie_name'])
    url1 = os.path.join(urlModel, 'model_recomendation.pkl')    
    gen_md,indices,cosine_sim,titles,smd = joblib.load(url1)
    
    url2 = os.path.join(urlModel, 'qualified.pkl')    
    modelQualification = joblib.load(url2)
    # print(gen_md.head(2))
    # print("RANKING DE PELICULA SEGUN GENERO")
    # print(build_chart('Comedy',gen_md).head(5))
    # print()
    # print("RANKING DE PELICULA SEGUN TITULO")
    # print(get_recommendations('The Godfather',indices,cosine_sim,titles).head(5))
    # print()
    # print("RANKING DE PELICULA SEGUN TITULO ORDENADO POR CALIFICACION")
    # print(improved_recommendations('The Godfather',indices,cosine_sim,smd).head(5))
    valor = 'The Godfather'
    response = {
        "ranked_items": build_chart('Comedy',gen_md).head(5).to_dict('records'),
        "recommendations_qualified": improved_recommendations(request.GET['movie_name'],indices,cosine_sim,smd).head(5).to_dict('records'),
    }
    return HttpResponse(json.dumps(response), content_type="application/json")

def home(request):    
    return render(request, 'chatbot/index.html')

def build_chart(genre,gen_md ,percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


def get_recommendations(title,indices,cosine_sim,titles):
    idx = indices[title] #se localiza el título
    sim_scores = list(enumerate(cosine_sim[idx])) #se realiza la semejanza de coseno con las películas
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #se ordenan en base a que tan parecido es (cerca a 1)
    sim_scores = sim_scores[1:31] #se sacan las 30 películas más cercanas
    movie_indices = [i[0] for i in sim_scores] #se guardan y se retornan los títulos de las 30 peliculas
    return titles.iloc[movie_indices]


def improved_recommendations(title,indices,cosine_sim,smd):
  #Se alcanza en la variable idx la info de la película por el título
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)  
      
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx])) #Se obtiene la similitud de coseno de la película, con todas las demás
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # se ordena las películas de acuerdo a su similitud
    sim_scores = sim_scores[1:26]# Solo se quedan las 25 primeras películas, sin contar la película inicial
    movie_indices = [i[0] for i in sim_scores]# Se obtiene los índices de las 25 películas
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']] #De esa lista de películas, se saca la información resultante
    #Se convierte en enteros los valores de las finas vote_counts y vote_average
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    #Se obtiene el promedio general de votos
    C = vote_averages.mean()
    #m es el mínimo de votos necesarios para aparecer en la lista de películas principales
    #para los votos mínimos necesarios, se usa el percentil  60 como punto de corte  
    m = vote_counts.quantile(0.60)
    #Se quedan en la variable qualified solo con las películas cuyo conteo de votos sean mayor a m.
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()).astype('int') & (movies['vote_average'].notnull()).astype('int')]
    # qualified['vote_count'] = qualified['vote_count'].astype('int')
    # qualified['vote_average'] = qualified['vote_average'].astype('int')
    #se crea una columna de calificación ponderada
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    #Se ordena de acurdo a la calificación ponderada
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    

    return qualified