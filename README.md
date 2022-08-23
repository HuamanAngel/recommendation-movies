# 📽 Sistema recomendador de películas 🎬

### Proyecto realizado por el grupo 6 para el curso de Software inteligentes para la creación de un chatbot basado en Question Answering utilizando dialogflow para la recepción de mensajes por parte del usuario final y python para el diseño del sistema recomendador de películas. 

La llegada de la pandemia ocasionó varios cambios en el estilo de vida de la población, siendo uno de ellos el del entretenimiento. Con el incremento del Streaming, las personas buscan recomendaciones sobre qué película piensan ver. Por ello, en este estudio se hizo la implementación de un Sistema Recomendador de películas utilizando procesamiento de lenguaje natural a través de Question Answer.


## Uso 

Primero arracamos levantamos el servidor (Framework Django), para esto nos 
ubicamos en la ruta raiz y ejecutamos los siguientes comandos : 

- python -m venv env
- env\Scripts\activate
- pip install -r requirements.txt
- python manage.py runserver

Ahora accedemos a la url http://localhost:8000/ y comprobamos que este correctamente levantado

El proyecto funciona en base a un modelo entrenado de recomendacion de peliculas, para 
generar este modelo podemos generarlo

La aplicacion funciona con un modelo entrenado, para generar este modelo tenemos
que descargar model_recomendation.pkl en el siguiente enlace [enlace](https://drive.google.com/drive/folders/1CeFqm3dvBcJvoAeMjbCmmA5XtldvW2iN?usp=sharing) o lo podemos generar desde el proyecto solo accediendo hacia la ruta '/training/'.

Hay 2 metodos que permiten la recomendacion : 
- build_chart : como 1er parametro se le pasa el genero, nos da como resultado una lista de peliculas ordenadas segun el promedio de votos
- improved_recommendations : Como 1er parameetro se pasa el nombre de la pelicula, nos da como resultado una lista de peliculas ordenadas segun el voto promedio

**USO DEL WEBHOOK CON DIALOGFLOW**

El metodo webhook dentro de la ruta 'webhook/' conecta con Diaglogflow, para su uso :
- Crearse una cuenta en Dialogfow
- Instalar ngrok y apuntar hacia el puerto 8000 ( o en el puerto abierto por django)
    - ngrok http 8000
- Ahora en fulfillment dentro de Dialogflow, colocamos la url que nos da ngrok y apuntamos a la ruta /webhook
    Ejemplo : 'https://xxxxxxxxxxxx.xx/webhook/'
- Ahora en entity creamos una llamada 'genero', aca registramos todos los generos que tiene el modelo
- Creamos dos intenciones :
    - Intencion de recomendar en base a genero : Creamos una intencion y colocamos como nombre de la accion         'recomendarGenero' y colcamos como parametro no requerido 'generos' ( lo enlazamos con la entidad 'genero' - @generos)
    
    - Intencion de recomendar basado en nombre : Creamos una intencion y colocamos como nombre de la accion         'recomendarNombre' y colcamos como parametro no requerido name_movie ( lo enalzamos con @system.any)

- Con esto ya deberia estar funcionando, escribir texto dentro del chatbot y listo.
## Datasets 🗂

En el desarrollo del sistema recomendador se utilizó los siguientes datasets:
- credits.csv
- final_movie_dataset.xlsx
- keywords.csv
- links.csv
- movies_metadata

Se pueden encontrar todos en el siguiente [enlace](https://drive.google.com/drive/folders/1c7ooUt2F5kgw3E6kJI8OhzH9Oi_JGE9c?usp=sharing).

## Metodología CRISP-DM

Se desarrolló la parte práctica del proyecto, donde se aplicó las fases de la metodología CRISP-DM, al dataset recolectado. 
Para la elaboración se utilizó la interfaz de Anaconda Navigator, en específico la aplicación de Jupyther Notebook.

Las fases implementadas a los datasets fueron 3:

Fase 1: Comprensión del negocio

- **Objetivo del negocio** : 
Sistema pueda recomendar películas usando Question Answer.

- **Criterios de éxito** : 
Permita mantener conversaciones y que vaya filtrando las respuestas en base a las respuestas del usuario
Se cuenta un dataset llamado “movie_metadata.xls” donde alberga 45.000 registros de películas, donde hay datos sobre popularidad, duración de película, votos ,etc. que son claves para la recomendación de películas.
Como lo que se busca es que se pueda recomendar películas, entonces se busca hacer un chatbot capaz de recomendar películas en base a la sinopsis, nombre de personajes, popularidad y mejores votados.

Fase 2: Compresión de los datos 

Dentro de los dataset utilizados son los siguientes:
- **movies_metadata.xls**: Es el archivo principal de metadatos de películas que contiene información de 45.000 películas, que son recolectadas de la página web MovieLens. Las características que contienen son carteles, fondos, presupuesto, ingresos, fechas de lanzamiento, idiomas, países de producción y empresas.
- **keywords.csv**: contiene las palabras clave de la trama de la película para nuestras películas MovieLens. Disponible en forma de un objeto JSON en cadena.
- **credits.csv**: contiene información sobre el reparto y el equipo técnico de todas nuestras películas. Disponible en forma de un objeto JSON en cadena.
- **links.csv**: el archivo que contiene los ID de TMDB e IMDB de todas las películas que aparecen en el conjunto de datos de Full MovieLens.

![](https://i.postimg.cc/rpybwF2s/Imagen1.png)

Fase 3: Preparación de datos

Se juntó todos los .csv dentro de uno solo para facilitar su uso, cada uno de estos dataset tiene un ID de referencia hacia movies_data.xls, entonces lo que se hizo es juntar los datos a través del índice.

![](https://i.postimg.cc/SxTm7PrZ/Imagen2.png)

Se removieron las columnas que no necesitamos, incluyendo también las columnas que son identificadores de películas que se usó para juntar entre los dataset. Además se **normalizaron** los tipos de datos para poder trabajarlo en las estadísticas.

![](https://i.postimg.cc/Wbd81Fyh/IMAGEN3.png)

## Modelamiento ⚙

Se divide en 2, sistema recomendador simple y basado en contenido.
- **Sistema recomendador simple**:

Se realiza una importación de los datos que usaremos, este tiene una extensión .xlsx y por nombre “final_movie_dataset”. Luego se procede a obtener las películas más votadas y se le añade una calificación a cada una  con respecto al conteo de votos y promedio de calificación por película.

- **Sistema recomendador basado en contenido**:

En este caso usaremos los datos del .csv con nombre “links_small”. Se realiza un filtrado para obtener los datos más relevantes como es el ID y el título de la película. 
Esta nueva matriz se combinará con un nuevo dataset que usaremos llamado “credits.csv”, el cual nos entrega más datos de la película como actores, directores y otros más.
Se procede a añadir más datos a nuestra matriz que contenía solo IDs y título de la película, en este caso ahora tendrá su director, elenco, y actores (3) respectivos. 

Se realiza una extracción de las palabras clave para crear un nuevo campo dentro de nuestra matriz final, para esto se hace varias operaciones descartando, contándose, devolviendo a su raíz para obtener al final una variable sólida respectiva a una película en concreto junto a reparto, director y género.

El resultado final del sistema son los modelos de películas en base a su calificación y su género, estos tienen de nombre “qualified.pkl” y “model_recomendation.pkl” respectivamente. 

Se puede visualizar los modelos en siguiente [enlace](https://drive.google.com/drive/folders/1CeFqm3dvBcJvoAeMjbCmmA5XtldvW2iN?usp=sharing).

## Implementación 

Dialogflow nos entrega un flujo simple de entregas para integraciones.

![image](https://user-images.githubusercontent.com/55029565/186072495-e481f1be-40d1-4409-8041-77e9e5234bbf.png)

Este flujo nos muestra como es el funcionamiento básico de nuestro chatbot empezando por los datos ingresados por el usuario final, pasando por Dialogflow que nos entrega un intent y los parámetros coincidentes a la expresión que mediante un webhook enlazaremos a nuestra API y terminando con un mensaje que satisfaga la necesidad del usuario.

Para el uso del webhook en Dialogflow es necesario usar el enlace que nos brinda ngrok para correr nuestro Sistema recomendador y este pueda recibir los datos del usuario.

El usuario interactúa con nuestro chatbot con la finalidad de encontrar tal vez el nombre de una película, actor, director, mejor película de un género en específico, etc.

Nuestro sistema en base a los datos del usuario recogidos por Dialogflow brinda la mejor respuesta posible. Esto se refiere por ejemplo a una petición del usuario dada de la siguiente manera: “Quiero ver la mejor película de terror”, donde el sistema tomará como parámetro el género terror y responderá con la película con mayor puntaje, según nuestro sistema, del género solicitado.


