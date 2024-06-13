# Limpieza de Datos con R (Data Cleaning with R)

- 👋 Hi, I’m @Cesarandres91
- 👀 I’m interested in data quality, data science and front end development
- 🌱 I’m currently learning data science.
- 💞️ I’m looking to collaborate on data quality proyects.
- 📫 How to reach me https://www.linkedin.com/in/andreschile/

#Predicción de la fuga de clientes en una empresa de telecomunicaciones

Contexto: Se proporciona el conjunto de datos churn-analysis.csv, el cual consiste en
identicar a los clientes que fugaron de la compañía.
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

##Desarrollo
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

#Versión en R:
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
#Versión en Python:
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


#Versión en R:
```Code R
glimpse(df)
skim(df)
```

#Versión en Python:
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

###Versión en R:
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

###Versión en Python:
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

##Analizaremos la Correlación de variables

###Versión en R:
```Code R
# Identificar las columnas numéricas en el DataFrame
numeric.var <- sapply(df, is.numeric)

# Calcular la matriz de correlación solo para las columnas numéricas
corr.matrix <- cor(df[, numeric.var])

# Graficar la matriz de correlación usando corrplot
corrplot(corr.matrix, main="\n\nGráfica de correlación para variables numéricas", method="number")
```
###Versión en Python:
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
###Versión en R:
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
###Versión en Python:
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
Dado la información anterior ahora se procederá a realizar las transformaciones necesarias sobre la data se pudo identificar al menos 3 tareas a realizar, se debe trabajar los outliers en las variables númericas, se debe suavizar el desbalanceo de la variable churn y se debe tomar la decisión si prescindir o no de las variables correlacionadas.

<!---
Cesarandres91/Cesarandres91 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
