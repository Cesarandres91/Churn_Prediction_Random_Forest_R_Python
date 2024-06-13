# Limpieza de Datos con R (Data Cleaning with R)

- 👋 Hi, I’m @Cesarandres91
- 👀 I’m interested in data quality, data science and front end development
- 🌱 I’m currently learning data science.
- 💞️ I’m looking to collaborate on data quality proyects.
- 📫 How to reach me https://www.linkedin.com/in/andreschile/

# Predicción de la fuga de clientes en una empresa de telecomunicaciones
Desarrollaremos un modelo predictivo utilizando el conjunto de datos "churn-analysis.csv", el cual proporciona información sobre los clientes que abandonaron la compañía.

Las descripciones de las columnas de la base de datos son:
state: Region del usuario.
area.code: Codigo de area.
phone.number: Numero telefonico.
international.plan: Plan internacional (yes o no).
voice.mail.plan: Plan con correo de voz (yes o no).
number.vmail.messages: Cantidad de mensajes virtuales posee.
total.day.minutes: Cantidad de minutos diarios.
total.day.calls: Cantidad de llamadas diarias.
total.day.charge: Cantidad del costo diario.
total.eve.minutes: Cantidad de minutos en la tarde.
total.eve.calls: Cantidad de llamadas en la tarde.
total.eve.charge: Cantidad de costo en la tarde.
total.night.minutes: Cantidad de minutos en la noche.
total.night.calls: Cantidad de llamadas en la noche.
total.night.charge: Cantidad de costo en la noche.
total.intl.minutes: Cantidad de minutos internacionales.
total.intl.calls: Cantidad de llamadas internacionales.
total.intl.charge: Cantidad de costo internacionales.
customer.service.calls: Cantidad de llamados a la mesa de ayuda
churn: Fuga del cliente (True o False).

## Desarrollo
Para la metodología de trabajo se usará como guía el modelo CRISP-DM que divide el proceso en fases: 
1) Comprensión del negocio
2) Comprensión de los datos
3) Preparación de los datos
4) Fase de Modelado
5) Evaluación e Implementación.

Además, para el análisis se utilizará el lenguaje R en Rstudio.

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/01d6ed26-bdd5-45cf-8e6f-41bec470aedf)

## 1 - Comprensión del negocio.
En este caso el contexto lo explica, el objetivo es la Predicción de la fuga de clientes en una empresa de telecomunicaciones dado el conjunto de datos churn-analysis.csv. Por lo cual nuestra variable objetivo será “churn” (variable respuesta) la cual muestra la Fuga del cliente (True o False) y para esto se realizará un análisis exploratorio de los datos para mejorar la comprensión de ellos y poder aplicar las transformaciones necesarias para luego implementarun modelo de Random forest en función de diversas variables de entrada (variables explicativas). El árbol de decisión es una representación para clasificar nuevos ejemplos y el aprendizaje basado en este tipo de árboles son una de las técnicas más eficaces para la clasificación supervisada.

## 2 - Comprensión de los datos.
Para esta etapa se realizará un análisis exploratorio de los datos utilizando el software R. Se leerá el documento usando read.csv() y luego para efectos prácticos en mi experiencia voy a desactivar la notación científica para entender mejor los números y transformar los nombres de las columnas para que todos los puntos sean guiones bajos y poder facilitar su uso, y luego ejecutar la función str() que nos permita conocer la estructura interna de cada una de las variables.

Podemos llevar a cabo esto con los siguientes códigos en R o en Python,

### Versión en R:
```Code R
# Leer el archivo CSV
df <- read.csv("C:/churn-analysis.csv", sep=";")

# Deshabilitar la notación científica en la salida
options(scipen=999)

# Renombrar las columnas reemplazando los puntos por guiones bajos
names(df) <- gsub('\\.','_',names(df))

# Mostrar la estructura del DataFrame
str(df)
```
### Versión en Python:
```Code python
import pandas as pd

# Leer el archivo CSV
df = pd.read_csv("C:/churn-analysis.csv", sep=";")

# Deshabilitar la notación científica en la salida
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Renombrar las columnas reemplazando los puntos por guiones bajos
df.columns = df.columns.str.replace('.', '_')

# Mostrar la estructura del DataFrame
print(df.info())
```
PD: Desactivar la notación científica hace que los números sean más fáciles de leer y entender, sin esos molestos exponentes. Cambiar los puntos por guiones bajos en los nombres de las columnas evita errores técnicos y hace tu código más limpio y legible. Así te aseguras de que todo funcione bien y sea más fácil de mantener, ¡sin dolores de cabeza!

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/3826fb7f-f60a-4b95-bd02-1dd842e047f5)

Dado lo anterior ejecutaré las funciones glimpse() y skim(), que entregan información muy útil, la primera mostrando el tipo de variable, indicando si son continuas (numéricas) o categóricas (factor) y da información general muy parecida a str(), luego la función skim() obtenemos el número de missing (valores faltantes), numero de datos completos, min, max, media, desviación estándar, información de los cuartiles y una pequeña gráfica que nos da un primer acercamiento sobre la distribución de los datos.


### Versión en R:
```Code R
glimpse(df)
skim(df)
```

### Versión en Python:
```Code python
import pandas as pd
from pandas_profiling import ProfileReport  

print(df.info())
    print(df.describe(include='all'))

# Generar el informe detallado similar a skim() usando pandas_profiling
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

# Mostrar el informe
profile.to_notebook_iframe()
```

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/7f78f098-356c-46b1-8836-7d380f17617c)

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/f61b706b-2c72-4ba6-8a73-38bececad90d)

Con skim obtenemos el número de missing (valores faltantes), numero de datos completos, media, desviación y información de los cuartiles.

##Primeras conclusiones:
- No hay valores nulos
- La variable phone_number tiene 3.333 registros únicos (misma cantidad de registros existentes) por lo que data la distribución de las demás variables no aportaría mayor valor al estudio y puede ser eliminada.
- Problemas con tipos de variables: state, area_code, , international_plan , voice_mail_plan y churn se comportan como variables del tipo factor y deben ser cambiadas.
- Viendo la información de los cuartiles variales como customer_service_calls podrían presentar valores outliers que afectarían la implementación correcta del modelo.

Removeré la columna pone_number, luego transformaré las variables mencionadas anteriormente al tipo factor y procederé a graficar por separado las variables del tipo factor y las del tipo númerico.

### Versión en R:
```Code R
# Eliminar la columna 'phone_number' del DataFrame
df$phone_number <- NULL

# Convertir varias columnas a factores (categorías)
df <- df %>%
  mutate(
    state = as.factor(state),
    area_code = as.factor(area_code),
    international_plan = as.factor(international_plan),
    voice_mail_plan = as.factor(voice_mail_plan),
    churn = as.factor(churn)
  )

# Seleccionar solo las columnas que son factores y reorganizar el DataFrame para visualización
df %>%
  select_if(is.factor) %>% # Seleccionar solo las columnas categóricas (factores)
  gather() %>% # Reorganizar el DataFrame de ancho a largo formato
  ggplot(aes(value)) + # Crear un gráfico de barras con 'ggplot2'
  geom_bar() + # Agregar barras al gráfico
  facet_wrap(~key, scales='free') + # Crear un gráfico de facetas, una para cada variable categórica
  theme(axis.text=element_text(size=6)) # Ajustar el tamaño del texto en los ejes

# Seleccionar solo las columnas que son numéricas y reorganizar el DataFrame para visualización
df %>%
  select_if(is.numeric) %>% # Seleccionar solo las columnas numéricas
  gather() %>% # Reorganizar el DataFrame de ancho a largo formato
  ggplot(aes(value)) + # Crear un gráfico de densidad con 'ggplot2'
  geom_density() + # Agregar densidades al gráfico
  facet_wrap(~key, scales='free') + # Crear un gráfico de facetas, una para cada variable numérica
  theme(axis.text=element_text(size=6)) # Ajustar el tamaño del texto en los ejes
```

### Versión en Python:
```Code python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv("C:/churn-analysis.csv", sep=";")

# Eliminar la columna 'phone_number'
df = df.drop(columns=['phone_number'])

# Convertir las columnas especificadas a tipo categórico
df['state'] = df['state'].astype('category')
df['area_code'] = df['area_code'].astype('category')
df['international_plan'] = df['international_plan'].astype('category')
df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
df['churn'] = df['churn'].astype('category')

# Visualización de las variables categóricas
categorical_cols = df.select_dtypes(include=['category']).columns
df_melted = df.melt(value_vars=categorical_cols)

g = sns.FacetGrid(df_melted, col='variable', col_wrap=3, sharex=False, sharey=False)
g.map(sns.countplot, 'value')
g.set_axis_labels("", "")
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, ha='right', fontsize=6)
plt.show()

# Visualización de las variables numéricas
numeric_cols = df.select_dtypes(include=['number']).columns
df_melted_numeric = df.melt(value_vars=numeric_cols)

g = sns.FacetGrid(df_melted_numeric, col='variable', col_wrap=3, sharex=False, sharey=False)
g.map(sns.kdeplot, 'value', fill=True)
g.set_axis_labels("", "")
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, ha='right', fontsize=6)
plt.show()
```
#Gráficas de distribución de variables del tipo factor
![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/0aaafd85-8455-4e61-9d48-771eea995a09)

Se puede ver que el area_code ampliamente mayoritario es el 415, la variable objetivo churn posee un desbalanceo alto hacia el valor “False”, la variable “international_plan” muestra un amplio dominio del “no”, la variable State tiene un valor máximo por sobre los 100 que se destaca sobre los demás valores, la variable voice_mail_plan muestra un amplio domino del “no”.

#Gráficas de distribución de variables del tipo factor
![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/44c24ae8-576b-41af-bdaf-c15267d0fa6f)

Casi todas las variables muestran una distribución normal excepto customer_service_calls (Cantidad de llamados a la mesa de ayuda), number_vmail_messages (Cantidad de mensajes virtuales posee) y total_intl_calls (Cantidad de llamadas internacionales).
Volvemos a realizar un skim() luego de la corrección del tipo de variables.

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/1bfe7ad8-4e1f-4086-b5dd-4d3550086690)

Vemos que el mayor state que se repite es WV, el código de área más repetido es 415 con casi el 50% de los registros, el 90% (3010) no corresponde a plan internacional, el 72% de los registros (2411) no corresponde a vocie_mail_plan y que efectivamente la variable churn se encuentra desbalanceada 86% a False y 14% a True, dado que lo ideal es que la variable objetivo tenga una proporción parecida o tener un mínimo cercano de 80% y 20% para que el modelo no prediga por defecto siempre la misma opción más repetida y en este caso señalando “False” ya que el modelo sería inservible.

## Analizaremos la Correlación de variables

### Versión en R:
```Code R
# Identificar las columnas numéricas en el DataFrame
numeric.var <- sapply(df, is.numeric)

# Calcular la matriz de correlación solo para las columnas numéricas
corr.matrix <- cor(df[, numeric.var])

# Graficar la matriz de correlación usando corrplot
corrplot(corr.matrix, main="\n\nGráfica de correlación para variables numéricas", method="number")
```
### Versión en Python:
```Code Python
# Identificar las columnas numéricas en el DataFrame
numeric_var = df.select_dtypes(include=[np.number])

# Calcular la matriz de correlación solo para las columnas numéricas
corr_matrix = numeric_var.corr()

# Graficar la matriz de correlación usando seaborn y matplotlib
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("\n\nGráfica de correlación para variables numéricas")
plt.show()
```

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/4e0ff8ef-83a9-48d8-8380-7ebd82ecddfd)

Vemos que existen 4 correlaciones altas entre variables:
o total_day_calls con total_day_minutes,
o total_eve_charge con total_eve_minutes,
o total_night_chargue con total_night_minutes
o total_intl_chargue con con total_intl_minutes.

Identificación de outliers,
### Versión en R:
```Code R
# Seleccionar solo las columnas numéricas del DataFrame
outliers <- df %>%
  select_if(is.numeric) %>%
  # Crear un diagrama de cajas y bigotes para cada variable numérica
  boxplot(df, main = "Diagrama de cajas y bigotes - todas las variables",
          outbg = "red", # Color de fondo para los outliers
          whiskcol = "blue", # Color de los 'bigotes'
          outpch = 25) # Tipo de punto para los outliers
```
### Versión en Python:
```Code Python
# Crear un diagrama de cajas y bigotes para cada variable numérica
plt.figure(figsize=(12, 8))  # Establecer el tamaño de la figura
sns.boxplot(data=df.select_dtypes(include=[np.number]),  # Seleccionar solo columnas numéricas
            palette="cool")  # Especificar una paleta de colores
            
plt.title("Diagrama de cajas y bigotes - todas las variables")  # Añadir título a la figura
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejor visualización
plt.grid(True)  # Añadir cuadrícula para facilitar la lectura
plt.show()  # Mostrar el gráfico
```
![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/5fa1c050-a724-4327-8db4-d8472307876a)

En la gráfica se han marcado como triángulos rojos todos los outliers existentes de cada variable númerica y como se puede apreciar todas contienen datos outliers tanto sobre tercer cuartil como bajo el primer cuartil.

## 3 - Preparación de los datos

A partir de la información analizada previamente, se identificaron tres tareas esenciales para la preparación de los datos:

a. **Tratamiento de Outliers en Variables Numéricas:**
   Los valores atípicos que excedan 1.5 veces el rango intercuartílico serán limitados. Específicamente, los valores por debajo del bigote inferior serán reemplazados por el percentil 5, y los valores por encima del bigote superior serán reemplazados por el percentil 95.

b. **Suavizar el Desbalance de la Variable `churn`:**
   Se aplicarán técnicas de balanceo de clases para mejorar la distribución de la variable objetivo `churn`, mejorando así la capacidad predictiva del modelo.

c. **Análisis de Variables Correlacionadas:**
   Se evaluará la correlación entre variables para decidir si es necesario descartar algunas con el fin de reducir la multicolinealidad y mejorar la interpretación del modelo.

### a - Tratamiento de Outliers
### Versión en R:
```Code R
# Definición de la función 'replace_outliers' para tratar outliers
replace_outliers <- function(x, removeNA = TRUE) {
    # Calcular los cuartiles 1 y 3, eliminando NA si se especifica
    qrts <- quantile(x, probs = c(0.25, 0.75), na.rm = removeNA)
    # Calcular los límites de capping usando el percentil 5 y 95
    caps <- quantile(x, probs = c(0.05, 0.95), na.rm = removeNA)
    # Calcular el rango intercuartílico (IQR)
    iqr <- qrts[2] - qrts[1]
    # Calcular el límite de los 'bigotes' para los diagramas de caja (1.5 * IQR)
    h <- 1.5 * iqr
    # Reemplazar valores más allá de 1.5*IQR por debajo del cuartil 1 con el percentil 5
    x[x < qrts[1] - h] <- caps[1]
    # Reemplazar valores más allá de 1.5*IQR por encima del cuartil 3 con el percentil 95
    x[x > qrts[2] + h] <- caps[2]
    return(x)
}

# Aplicar la función 'replace_outliers' a múltiples columnas del DataFrame 'df'
df$number_vmail_messages <- replace_outliers(df$number_vmail_messages)
df$total_day_minutes <- replace_outliers(df$total_day_minutes)
df$total_day_calls <- replace_outliers(df$total_day_calls)
df$total_day_charge <- replace_outliers(df$total_day_charge)
df$total_eve_minutes <- replace_outliers(df$total_eve_minutes)
df$total_eve_calls <- replace_outliers(df$total_eve_calls)
df$total_eve_charge <- replace_outliers(df$total_eve_charge)
df$total_night_minutes <- replace_outliers(df$total_night_minutes)
df$total_night_calls <- replace_outliers(df$total_night_calls)
df$total_night_charge <- replace_outliers(df$total_night_charge)
df$total_intl_minutes <- replace_outliers(df$total_intl_minutes)
df$total_intl_calls <- replace_outliers(df$total_intl_calls)
df$total_intl_charge <- replace_outliers(df$total_intl_charge)
df$customer_service_calls <- replace_outliers(df$customer_service_calls)
```
### Versión en Python:
```Code Python
import pandas as pd
import numpy as np

# Definición de la función para reemplazar outliers
def replace_outliers(series, remove_na=True):
    if remove_na:
        series = series.dropna()  # Opcional: eliminar valores NaN
    # Calcular los cuartiles y los límites para los outliers
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Percentiles para reemplazo
    lower_cap = series.quantile(0.05)
    upper_cap = series.quantile(0.95)
    # Reemplazar los outliers
    series = series.apply(lambda x: lower_cap if x < lower_bound else (upper_cap if x > upper_bound else x))
    return series

# Aplicación de la función a varias columnas del DataFrame 'df'
columns_to_fix = [
    'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
    'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 
    'total_eve_charge', 'total_night_minutes', 'total_night_calls',
    'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 
    'total_intl_charge', 'customer_service_calls'
]

for column in columns_to_fix:
    df[column] = replace_outliers(df[column])
```
Si obtenemos nuevamente la gráfica vemos que los outliers ya no se encuentran.

PD: Este enfoque es muy útil en análisis de datos para asegurar que los resultados no estén sesgados por valores atípicos extremos, y que el modelo de datos resultante sea más robusto y representativo del comportamiento "normal" esperado.

### b - Suavizar el Desbalance de la Variable
Se utilizará la técnica de undersampling (inframuestreo) y over sampling (sobremuestreo) de manera conjunta con la ayuda de la libería ROSE y la función ovun.sample donde el método es “both” señalando que utiliza ambas técnicas, buscando el balance 50/50 de la variable churn.

### Versión en R:
```Code R
# Balanceo del conjunto de datos usando sobremuestreo y submuestreo con la función ovun.sample
data_balanced_both <- ovun.sample(
  churn ~ .,      # Fórmula: la variable dependiente 'churn' y todas las variables independientes
  data = df,      # Conjunto de datos original
  method = "both",# Método de balanceo: ambos (sobremuestreo y submuestreo)
  p = 0.5,        # Probabilidad deseada para la clase minoritaria en el conjunto balanceado
  N = 3333,       # Número total de muestras en el conjunto balanceado
  seed = 1        # Semilla para la reproducibilidad de los resultados
)$data            # Extraer el conjunto de datos balanceado
```

### Versión en Python:
```Code Python

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Separar las características y la variable objetivo
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Crear un objeto SMOTETomek para el balanceo de clases
smote_tomek = SMOTETomek(random_state=1)

# Aplicar el balanceo de clases al conjunto de entrenamiento
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Crear un DataFrame del conjunto de datos balanceado
data_balanced_both = pd.concat([X_resampled, y_resampled], axis=1)
```
Utilizando skim() sobre este nuevo df comprabamos que ya se encuentra balanceado, con 1721 valores False y 1612 valores True.
![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/082dcc2d-b1e0-4def-b299-d2193f216c6d)

### Variables correlacionadas
Como revisamos anteriormente, existen cuatro correlaciones altas entre variables que tienen sentido lógico. Por ejemplo, la relación entre total_day_calls y total_day_minutes: a mayor número de llamadas durante el día, es probable que el total de minutos también sea mayor. Dado que aplicaremos un modelo de Random Forest, no descartaremos ninguna de estas variables correlacionadas. Sin embargo, será importante tener en cuenta estas correlaciones al momento de interpretar los resultados y sacar conclusiones.


4 - Fase de Modelado
Primero, separaremos el conjunto de datos (antes del balanceo mostrado anteriormente) en conjuntos de entrenamiento (70%) y prueba (30%) utilizando la librería caTools y la función sample.split, especificando un ratio de 0.7. Luego, aplicaremos nuevamente la corrección del desbalanceo en el conjunto de entrenamiento. Utilizaremos la función skim() para determinar la cantidad de datos en el conjunto de entrenamiento, y ese número se usará como parámetro N para la corrección del desbalanceo.

### Versión en R:
```Code R
# Establecer la semilla para reproducibilidad
set.seed(1991)

# Cargar la biblioteca caTools para la división del conjunto de datos
library(caTools)

# Dividir el conjunto de datos en entrenamiento (70%) y prueba (30%)
sample <- sample.split(df, SplitRatio = 0.7)
train <- subset(df, sample == TRUE)
test <- subset(df, sample == FALSE)

# Resumen del conjunto de entrenamiento
skim(train)

# Balancear el conjunto de entrenamiento utilizando la función ovun.sample
train_balanced <- ovun.sample(
  churn ~ .,       # Fórmula: la variable dependiente 'churn' y todas las variables independientes
  data = train,    # Conjunto de datos de entrenamiento
  method = "both", # Método de balanceo: ambos (sobremuestreo y submuestreo)
  p = 0.5,         # Probabilidad deseada para la clase minoritaria en el conjunto balanceado
  N = 2283,        # Número total de muestras en el conjunto balanceado
  seed = 1         # Semilla para la reproducibilidad de los resultados
)$data             # Extraer el conjunto de datos balanceado

# Cargar la biblioteca randomForest para el modelado
library(randomForest)

# Entrenar un modelo de Random Forest utilizando el conjunto de datos balanceado
modelo_bagging <- randomForest(
  churn ~ .,         # Fórmula: la variable dependiente 'churn' y todas las variables independientes
  data = train_balanced, # Conjunto de datos de entrenamiento balanceado
  mtry = 4,          # Número de variables a considerar en cada división de nodo
  importance = TRUE  # Calcular la importancia de las variables
)

# Graficar el modelo de Random Forest
plot(modelo_bagging)
```
### Versión en Python:
```Code Python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Establecer la semilla para reproducibilidad
np.random.seed(1991)

# Leer los datos, asumiendo que el archivo CSV ya está cargado en 'df'
# df = pd.read_csv("tu_archivo.csv")

# Separar las características y la variable objetivo
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir el conjunto de datos en conjuntos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Resumen del conjunto de entrenamiento
print(X_train.describe())
print(y_train.value_counts())

# Balancear el conjunto de entrenamiento utilizando SMOTE y Tomek links
smote_tomek = SMOTETomek(random_state=1)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# Crear un DataFrame del conjunto de datos balanceado (opcional, solo para visualización)
train_balanced = pd.concat([X_train_balanced, y_train_balanced], axis=1)

# Entrenar un modelo de Random Forest utilizando el conjunto de datos balanceado
modelo_bagging = RandomForestClassifier(n_estimators=100, max_features=4, random_state=1, oob_score=True)
modelo_bagging.fit(X_train_balanced, y_train_balanced)

# Graficar la importancia de las características
importances = modelo_bagging.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualizar la importancia de las características
plt.figure(figsize=(12, 6))
plt.title("Importancia de las características")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Evaluar el modelo con el conjunto de prueba
y_pred = modelo_bagging.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Ahora aplicaremos el modelo de Random Forest (bagging), tomando en cuenta los hiperparámetros y prestando especial atención a la importancia de las variables:

- ntree: Número de árboles en el bosque. Queremos estabilizar el error, pero usar demasiados árboles puede ser ineficiente.
- mtry: Número de variables aleatorias como candidatas en cada ramificación.
- sampsize: Número de muestras sobre las cuales entrenar. El valor por defecto es el 63.25%. Valores más bajos podrían introducir sesgo y reducir el tiempo de entrenamiento. Valores más altos podrían incrementar el rendimiento del modelo pero a riesgo de causar overfitting. Generalmente se mantiene en el rango del 60-80%.
- nodesize: Mínimo número de muestras dentro de los nodos terminales. Esto equilibra el bias y la varianza.
- maxnodes: Número máximo de nodos terminales.


### Versión en R:
```Code R
  # Entrenar un modelo de Random Forest utilizando el conjunto de datos de entrenamiento
modelo_bagging <- randomForest(
  churn ~ .,         # Fórmula: la variable dependiente 'churn' y todas las variables independientes
  data = train,      # Conjunto de datos de entrenamiento
  mtry = 4,          # Número de variables aleatorias a considerar en cada división de nodo
  importance = TRUE  # Calcular la importancia de las variables
)

# Graficar el error del modelo de Random Forest a medida que se agregan más árboles
plot(modelo_bagging)

# Graficar la importancia de las variables en el modelo de Random Forest
varImpPlot(modelo_bagging)

# Graficar nuevamente el error del modelo de Random Forest con color personalizado
plot(modelo_bagging, col = "firebrick")
```
### Versión en Python:
```Code Python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Separar las características y la variable objetivo
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir el conjunto de datos en conjuntos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Entrenar un modelo de Random Forest utilizando el conjunto de datos de entrenamiento
modelo_bagging = RandomForestClassifier(n_estimators=100, max_features=4, random_state=1, oob_score=True)
modelo_bagging.fit(X_train, y_train)

# Graficar el error del modelo de Random Forest a medida que se agregan más árboles
oob_error = 1 - modelo_bagging.oob_score_
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(modelo_bagging.estimators_) + 1), [1 - est.oob_score_ for est in modelo_bagging.estimators_], color='firebrick')
plt.xlabel('Número de árboles')
plt.ylabel('Error OOB')
plt.title('Error OOB vs. Número de árboles en el Random Forest')
plt.show()

# Graficar la importancia de las variables en el modelo de Random Forest
importances = modelo_bagging.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Importancia de las características')
plt.bar(range(X_train.shape[1]), importances[indices], align='center', color='firebrick')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# Evaluar el modelo con el conjunto de prueba
y_pred = modelo_bagging.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/424a2da3-9c5f-4adc-a793-62630a075284)

MeanDecreaseAccuracy mide cuánto reduce el error de clasificación la inclusión de este predictor en el modelo. En otras palabras, evalúa la importancia de una variable al observar el aumento en el error de clasificación cuando se elimina dicha variable del modelo.

MeanDecreaseGini mide la ganancia promedio de pureza por divisiones de una variable dada. Cuando una variable es útil, tiende a dividir los nodos mixtos (contienen múltiples clases) en nodos puros (contienen solo una clase). Esto indica que una variable predictora en particular juega un papel significativo en la partición de los datos en las clases definidas, aumentando así la pureza de los nodos en el árbol de decisión.

Aquí se observa que, aunque la variable State tiene un alto valor de Gini, su contribución a la precisión del modelo es baja. Por otro lado, las variables total_day_charge, total_day_minutes y customer_service_calls se destacan como determinantes en ambas medidas. De manera similar, aunque en menor medida, international_plan y total_eve_charge también juegan un papel importante en la precisión y la pureza del modelo.

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/63660403-c541-4baa-9e71-465963beb4f5)

Podemos observar que, a partir del árbol número 25 aproximadamente, el error comienza a estabilizarse.

## 5 - Evaluación e Implementación
Finalmente probaremos el modelo buscando predecir los datos de test y ver la perfomance de este.

### Versión en R:
```Code R
# Realizar predicciones en el conjunto de datos de prueba utilizando el modelo de Random Forest entrenado
pred <- predict(modelo_bagging, test, type = "class")

# Generar una tabla de confusión para comparar las predicciones con los valores reales
table(
  test[,"churn"], # Valores reales de la variable objetivo en el conjunto de prueba
  pred,           # Valores predichos por el modelo
  dnn = c("Actual", "Predicho") # Nombres para las dimensiones de la tabla (actuales y predichos)
)
```

### Versión en Python:
```Code Python
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Realizar predicciones en el conjunto de datos de prueba utilizando el modelo de Random Forest entrenado
pred = modelo_bagging.predict(X_test)

# Generar una tabla de confusión para comparar las predicciones con los valores reales
conf_matrix = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generar un informe de clasificación para evaluar el desempeño del modelo
class_report = classification_report(y_test, pred, target_names=['No Churn', 'Churn'])
print("\nClassification Report:")
print(class_report)
```

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/17cbd3d7-6954-4a6d-b35f-0a9cb19277eb)

Podemos observar que el modelo acertó en 884 casos de "No Churn" y en 125 casos de "Churn", sumando un total de 1009 aciertos. Por otro lado, el modelo falló en 29 casos (predicciones falsas de "Churn") y en 15 casos (predicciones falsas de "No Churn"), sumando un total de 44 errores.

Para evaluar aún más el desempeño del modelo, utilizaremos la curva ROC y su área bajo la curva (AUC), la curva lift, y la curva de sensibilidad/especificidad. Para ello, utilizaremos la librería ROCR junto con las funciones predict(), prediction(), y performance().

### Detalles de las métricas de evaluación:

- La curva ROC (Receiver Operating Characteristic) es una gráfica que muestra la capacidad de un modelo para distinguir entre clases. La AUC (Área Bajo la Curva) proporciona una única medida del rendimiento del modelo, donde un valor más alto indica un mejor rendimiento.
Curva Lift:

-  La curva lift muestra el rendimiento de un modelo predictivo en comparación con una selección aleatoria. Es útil para evaluar la efectividad del modelo en tareas de clasificación binaria, especialmente en el contexto de campañas de marketing.
Curva de Sensibilidad/Especificidad:

- La sensibilidad (recall) mide la proporción de verdaderos positivos correctamente identificados, mientras que la especificidad mide la proporción de verdaderos negativos. Estas curvas ayudan a entender el trade-off entre sensibilidad y especificidad para diferentes umbrales de decisión.


### Versión en R:
```Code R
# Realizar predicciones probabilísticas en el conjunto de datos de prueba utilizando el modelo de Random Forest entrenado
proba <- predict(modelo_bagging, test, type = "prob")

# Crear un objeto de predicción para ROCR utilizando las probabilidades de la clase positiva (churn)
predi_roc <- prediction(proba[,2], test[,"churn"])

# Calcular la curva ROC (True Positive Rate vs. False Positive Rate)
performance_rf <- performance(predi_roc, "tpr", "fpr")

# Graficar la curva ROC
plot(performance_rf, main = "Curva ROC")

# Calcular el AUC (Área Bajo la Curva) de la curva ROC
AUC <- performance(predi_roc, measure = "auc")

# Extraer el valor de AUC
AUCaltura <- AUC@y.values
AUCaltura

# Calcular la curva Lift (tasa de levantamiento)
perf2 <- performance(predi_roc, "lift", "rpp")

# Graficar la curva Lift
plot(perf2, main = "Curva LIFT", colorize = TRUE)

# Calcular la curva de sensibilidad/especificidad
perf3 <- performance(predi_roc, "sens", "spec")

# Graficar la curva de sensibilidad/especificidad
plot(perf3, main = "Curva de sensibilidad/especificidad")

```


### Versión en Python:
```Code Python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Realizar predicciones probabilísticas en el conjunto de datos de prueba utilizando el modelo de Random Forest entrenado
proba = modelo_bagging.predict_proba(X_test)[:, 1]

# Calcular la curva ROC y el AUC
fpr, tpr, _ = roc_curve(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)

# Graficar la curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Calcular la curva Lift
precision, recall, thresholds = precision_recall_curve(y_test, proba)
lift = precision / (np.sum(y_test) / len(y_test))

# Graficar la curva Lift
plt.figure(figsize=(10, 6))
plt.plot(thresholds, lift[:-1], color='darkorange', lw=2)
plt.xlabel('Umbral')
plt.ylabel('Lift')
plt.title('Curva Lift')
plt.show()

# Calcular la curva de Sensibilidad/Especificidad
plt.figure(figsize=(10, 6))
plt.plot(1 - fpr, tpr, color='darkorange', lw=2)
plt.xlabel('Especificidad')
plt.ylabel('Sensibilidad')
plt.title('Curva de Sensibilidad/Especificidad')
plt.show()
```
### Curva ROC
![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/43006478-5f95-4da9-b9a9-c6217e88c6d5)

Observaciones:

- Inicio de la Curva: La curva comienza en el punto (0,0), lo que indica que inicialmente no hay falsos positivos ni verdaderos positivos.

- Segmento Inicial de la Curva: La curva se eleva rápidamente hacia la esquina superior izquierda, lo cual es una característica de un buen modelo. Esto muestra que el modelo logra una alta tasa de verdaderos positivos mientras mantiene una baja tasa de falsos positivos en los primeros umbrales.

- Comportamiento General: La curva se mantiene cerca del eje Y y la parte superior del gráfico, lo que indica un alto rendimiento del modelo. Una curva ROC que se aproxima a la diagonal (la línea de no discriminación) sugiere un modelo que no discrimina bien entre las clases.

Junto con la gráfica ROC, obtenemos un valor del área bajo la curva (AUC) de 0.9249. Este alto valor de AUC indica que el modelo es un excelente predictor para identificar a los clientes que se fugan de la compañía.

### Curva LIFT
![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/a94d3624-c866-4963-81e8-c0296618e68d)

Observaciones:

- Inicio de la Curva: La curva comienza con un Lift alto alrededor de 7, indicando que las primeras predicciones del modelo son significativamente mejores que el azar.

- Comportamiento General: La curva desciende gradualmente, lo que es esperado a medida que se consideran más predicciones. Esto indica que el beneficio del modelo sobre una selección aleatoria disminuye pero sigue siendo superior.

- Segmento Final: Hacia el final, el Lift se aproxima a 1, lo que significa que el modelo ya no es mucho mejor que el azar cuando se predice una alta proporción de positivos.
Conclusión

El modelo es muy eficaz en sus primeras predicciones y mantiene una ventaja significativa sobre el azar hasta que se consideran la mayoría de las predicciones, es decir la curva Lift indica que el modelo es efectivo.

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/95919066-6246-46e5-aa30-288af45c9ce8)

Observaciones:

- Inicio de la Curva: La curva comienza cerca del punto (0, 1), lo que indica que al inicio se tiene una alta sensibilidad (casi todos los positivos son identificados) con una baja especificidad (casi todos los negativos no son identificados correctamente).

- Comportamiento General: A medida que aumenta la especificidad, la sensibilidad disminuye gradualmente. Esto es esperado, ya que hay un trade-off natural entre sensibilidad y especificidad.

- Segmento Plano: La curva es bastante plana al inicio, lo que indica que se puede mantener una alta sensibilidad sin sacrificar demasiada especificidad. Este comportamiento es deseable, ya que muestra que el modelo es capaz de identificar correctamente los positivos sin muchos falsos negativos.
  
- Caída al Final: Hacia el final de la curva, hay una caída abrupta. Este comportamiento sugiere que, en algún punto, el incremento de la especificidad empieza a sacrificar significativamente la sensibilidad, indicando que el modelo empieza a identificar menos casos positivos correctamente.
  
La curva muestra un buen desempeño del modelo, manteniendo una alta sensibilidad mientras la especificidad va aumentando gradualmente.
Trade-off Sensibilidad/Especificidad:La curva ilustra claramente el trade-off entre sensibilidad y especificidad, indicando que al incrementar una, se puede reducir la otra.

En resumen, esta curva de sensibilidad/especificidad sugiere que el modelo es bastante eficaz en balancear la identificación correcta de casos positivos y negativos, con un buen rendimiento global. La caída abrupta hacia el final es un indicador de que, a partir de cierto punto, mejorar la especificidad tiene un alto costo en términos de sensibilidad.

## Conclusión

El modelo permite una buena predicción con un 92.4% de desempeño, catalogando correctamente a los clientes que se fugan de la compañía. Dado que la variable churn inicialmente tenía una proporción de 86/14, este resultado indica que el modelo es muy efectivo. Este alto rendimiento fue posible gracias a un tratamiento cuidadoso de los outliers y la corrección del desbalanceo de los datos antes de entrenar el modelo.

Las evaluaciones de rendimiento mediante las curvas ROC y Lift validan que el modelo es un buen predictor:

Curva ROC: La curva ROC muestra que el modelo tiene una alta capacidad de discriminación, manteniendo una alta tasa de verdaderos positivos (sensibilidad) con una baja tasa de falsos positivos. Esto se refleja en un AUC de 0.9249, lo cual es un excelente indicador de rendimiento.

Curva Lift: La curva Lift muestra que el modelo tiene un alto valor de elevación en sus primeras predicciones, indicando que es significativamente mejor que el azar al identificar correctamente los casos positivos (clientes que se fugan). Aunque la ventaja del modelo disminuye a medida que se consideran más predicciones, sigue siendo superior a una selección aleatoria.

En resumen, el modelo es muy efectivo para predecir la fuga de clientes, y estos resultados sugieren que las técnicas de tratamiento de outliers y balanceo de datos fueron fundamentales para lograr un buen desempeño. Sin embargo, esto no implica que no se puedan mejorar los resultados con otros modelos. Sería interesante validar la eliminación de variables con alta correlación y explorar otros algoritmos de clasificación para potencialmente mejorar aún más el rendimiento.












<!---
Cesarandres91/Cesarandres91 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
