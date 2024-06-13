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

Desarrollo
Para la metodología de trabajo se usará como guía el modelo CRISP-DM que divide el proceso en fases: 
1) Comprensión del negocio
2) Comprensión de los datos
3) Preparación de los datos
4) Fase de Modelado
5) Evaluación e Implementación.

Además, para el análisis se utilizará el lenguaje R en Rstudio.

![image](https://github.com/Cesarandres91/DS_Data_cleansing_with_R/assets/102868086/01d6ed26-bdd5-45cf-8e6f-41bec470aedf)

1 - Comprensión del negocio.
En este caso el contexto lo explica, el objetivo es la Predicción de la fuga de clientes en una empresa de telecomunicaciones dado el conjunto de datos churn-analysis.csv. Por lo cual nuestra variable objetivo será “churn” (variable respuesta) la cual muestra la Fuga del cliente (True o False) y para esto se realizará un análisis exploratorio de los datos para mejorar la comprensión de ellos y poder aplicar las transformaciones necesarias para luego implementarun modelo de Random forest en función de diversas variables de entrada (variables explicativas). El árbol de decisión es una representación para clasificar nuevos ejemplos y el aprendizaje basado en este tipo de árboles son una de las técnicas más eficaces para la clasificación supervisada.

2 - Comprensión de los datos.
Para esta etapa se realizará un análisis exploratorio de los datos utilizando el software R. Se leerá el documento usando read.csv() y luego para efectos prácticos en mi experiencia voy a desactivar la notación científica para entender mejor los números y transformar los nombres de las columnas para que todos los puntos sean guiones bajos y poder facilitar su uso, y luego ejecutar la función str() que nos permita conocer la estructura interna de cada una de las variables.

Code R:
\`\`\`r
df <- read.csv("C:/churn-analysis.csv", sep=";")
options(scipen=999)
names(df) <- gsub('\\.','_',names(df))
str(df)
\`\`\`
<!---
Cesarandres91/Cesarandres91 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
