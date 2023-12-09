"""
Created on Thu Dec  7 00:45:42 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""
#%% LIBRERÍAS

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from RandomForest import RandomForest  

#%% CARGAR DATOS

breast_cancer = load_breast_cancer()
data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
data["label"] = breast_cancer.target
cols = ['worst concave points', 'worst area', 'mean concavity', 'mean compactness', 'label']   
data = data.loc[:, cols]
# Crear índices para la partición (80% entrenamiento, 20% prueba)
indices_particion = train_test_split(range(len(data)), test_size=0.2, random_state=13)

# Separar los datos en conjuntos de entrenamiento y prueba
datos_entrenamiento = data.loc[indices_particion[0]]
datos_prueba = data.loc[indices_particion[1]]

#%% cuántos árboles necesitamos en nuestro bosque?

num_arboles_list = [11, 21, 31, 41, 55, 61, 71, 91, 101]

# Lista para almacenar accuracies
accuracies = []

# Iterar sobre diferentes cantidades de árboles
for num_arboles in num_arboles_list:
    # Crear y entrenar bosque aleatorio
    forest = RandomForest(n_trees=num_arboles, max_depth=3, min_samples=5, random_state=1989)
    forest.fit(datos_entrenamiento)
    
    # Realizar predicciones
    predicted = forest.predict(datos_prueba)
    
    # Calcular accuracy y guardar en la lista
    accuracy = accuracy_score(predicted, datos_prueba["label"])
    accuracies.append(accuracy)
    
    # Imprimir resultados
    print(f"Number of Trees: {num_arboles}, Accuracy: {accuracy}")

# Gráficas
plt.plot(num_arboles_list, accuracies, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Trees in Random Forest')
plt.show()

#%% Cuántas características necesitamos?

# Configuración inicial
num_arboles = 21
porcentajes_caracteristicas = [0.1, 0.5, 0.8, 1]

# Listas para almacenar resultados
accuracies = []

# Iterar sobre diferentes porcentajes de características
for porcentaje in porcentajes_caracteristicas:
    # Crear bosque aleatorio con el porcentaje de características actual
    forest = RandomForest(n_trees=num_arboles, max_depth=3, min_samples=5, random_state=1989)
    forest.fit(datos_entrenamiento, max_features=porcentaje)
    
    # Realizar predicciones
    predicted = forest.predict(datos_prueba)
    
    # Calcular accuracy y guardar en la lista
    accuracy = accuracy_score(predicted, datos_prueba["label"])
    accuracies.append(accuracy)
    
    # Imprimir resultados
    print(f"Porcentaje de Características: {porcentaje}, Accuracy: {accuracy}")

# Graficar resultados
plt.plot(porcentajes_caracteristicas, accuracies, marker='o')
plt.xlabel('Porcentaje de Características')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Porcentaje de Características en el Bosque Aleatorio')
plt.show()

#%% Criterio de paro

# Configuración inicial
num_arboles = 21
porcentaje_caracteristicas = 0.5
min_samples_valores = [1, 3, 5, 8, 10]

# Listas para almacenar resultados
accuracies = []

# Iterar sobre diferentes valores de min_samples
for min_samples in min_samples_valores:
    # Crear bosque aleatorio con el valor de min_samples actual
    forest = RandomForest(n_trees=num_arboles, max_depth=3, min_samples=min_samples, random_state=1989)
    forest.fit(datos_entrenamiento, max_features=porcentaje_caracteristicas)
    
    # Realizar predicciones
    predicted = forest.predict(datos_prueba)
    
    # Calcular accuracy y guardar en la lista
    accuracy = accuracy_score(predicted, datos_prueba["label"])
    accuracies.append(accuracy)
    
    # Imprimir resultados
    print(f"Número Mínimo de Elementos en una Hoja: {min_samples}, Accuracy: {accuracy}")

# Graficar resultados
plt.plot(min_samples_valores, accuracies, marker='o')
plt.xlabel('Número Mínimo de Elementos en una Hoja')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Número Mínimo de Elementos en una Hoja en el Bosque Aleatorio')
plt.show()

#%% importancia de características

forest = RandomForest(n_trees=21, max_depth=3, min_samples=5, random_state=1989)
forest.fit(datos_entrenamiento, max_features=0.5)

# Acceder a las importancias de las características
feature_importances = forest.feature_importances

# Crear un gráfico de barras
features = list(feature_importances.keys())
importances = list(feature_importances.values())

plt.figure(figsize=(10, 6))
plt.bar(features, importances)
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.title('Importancia de las Características en el Bosque Aleatorio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#%% IMPRIMIR ÁRBOLES

forest.print_trees(n=10, random_state=3)

#%% EVALUACIÓN

# Obtener los índices utilizados durante el entrenamiento
training_indices = forest.get_used_indices()

# Obtener los índices no utilizados
unused_indices = set(datos_entrenamiento.index) - set(training_indices)

# Filtrar los datos no utilizados
datos_no_utilizados = datos_entrenamiento.loc[unused_indices]


predicted = forest.predict(datos_prueba)
# Matriz de Confusión y Accuracy
conf_matrix = confusion_matrix(predicted, datos_prueba["label"])
conf_matrix_table = pd.DataFrame(conf_matrix, columns=[0, 1], index=[0, 1])
accuracy = accuracy_score(predicted, datos_prueba["label"])

# Imprimir la matriz de confusión y el accuracy
print(conf_matrix_table)
print("Accuracy:", accuracy)

