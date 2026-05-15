# Evaluación offline de sistemas de recomendación

## Inteligencia de Negocio y en la Web

*Javier Francisco Dibo Gómez*

---

## Resumen

Se presenta una evaluación offline exhaustiva de sistemas de recomendación sobre el conjunto de datos rs-movie-cour, una variante reducida de MovieLens compuesta por 100 películas y 5.563 usuarios. Se comparan dieciséis configuraciones de tres familias de métodos: filtrado colaborativo basado en usuarios (UserKNN), filtrado colaborativo basado en ítems (ItemKNN), factorización matricial (SVD) y un sistema de recomendación basado en contenido mediante TF-IDF sobre etiquetas. La evaluación emplea validación cruzada de 5 particiones con las divisiones proporcionadas y mide el error de predicción (MAE, RMSE) y la calidad del ranking (Precision\@N, Recall\@N, F1\@N, NDCG\@N) para N ∈ {3, 5, 10} con umbral de relevancia 3,5. SVD con 30 factores latentes obtiene los mejores resultados en todas las métricas (MAE = 0,6195; F1\@10 = 0,3618; NDCG\@10 = 0,4530), seguido por las variantes SVD con menos factores y los métodos KNN con similitud de Pearson.

---

## 1. Introducción

Los sistemas de recomendación son herramientas fundamentales para personalizar la experiencia del usuario en plataformas de contenido digital. Su evaluación rigurosa requiere comparar distintos métodos bajo condiciones controladas mediante métricas que capturen tanto la precisión en la predicción de valoraciones como la calidad de las listas de recomendación generadas.

El presente trabajo evalúa de forma comparativa métodos de filtrado colaborativo —tanto basados en vecinos como en factorización matricial— y un recomendador de contenido basado en TF-IDF, desarrollado en la Práctica 1. Para cada configuración se calculan métricas de predicción de rating y de ranking sobre un mismo conjunto de datos de películas, utilizando validación cruzada estratificada de 5 particiones para garantizar la fiabilidad estadística de los resultados.

---

## 2. Conjunto de datos

El conjunto de datos rs-movie-cour es una variante reducida del dataset MovieLens con cuatro ficheros CSV. El fichero `movie-titles.csv` contiene 100 películas identificadas por código numérico y título con año de estreno. El fichero `users.csv` recoge 5.563 usuarios. El fichero `ratings.csv` almacena 338.355 valoraciones en escala 0,5–5,0. El fichero `movie-tags.csv` contiene 41.980 asignaciones de etiquetas, correspondientes a 4.209 etiquetas únicas.

La valoración media global es 3,66 (desviación típica 1,07), con predisposición positiva: el 4,0 es la valoración más frecuente (23,0 %), seguida del 5,0 (19,3 %). Cada usuario ha emitido una media de 60,81 valoraciones (mediana 61, desviación 25,42). The Matrix (1999) es la película más valorada, con 4.942 entradas y media de 4,30.

En cuanto a las etiquetas, la media de asignaciones por película es 419,8, con un máximo de 1.533 en Pulp Fiction (1994). Las etiquetas más frecuentes son "twist ending" (736), "sci-fi" (651) y "psychology" (617), lo que refleja una preferencia por el cine especulativo y de narrativa compleja. El vocabulario mezcla géneros, atributos narrativos, referencias a directores y criterios metacinematográficos, lo que enriquece pero también introduce ruido en los perfiles de ítem.

---

## 3. Métodos de recomendación

Se evalúan dieciséis configuraciones pertenecientes a cuatro familias. El método SVD++ queda excluido por incompatibilidad de la biblioteca scikit-surprise con la versión de Python utilizada (3.14).

### 3.1 Filtrado colaborativo basado en usuarios (UserKNN)

KNNWithMeans con corrección de media, k ∈ {10, 20, 30} y similitud ∈ {coseno, Pearson}: 6 configuraciones.

| Configuración | k | Similitud |
|---|---|---|
| UserKNN k=10 cosine | 10 | Coseno |
| UserKNN k=10 pearson | 10 | Pearson |
| UserKNN k=20 cosine | 20 | Coseno |
| UserKNN k=20 pearson | 20 | Pearson |
| UserKNN k=30 cosine | 30 | Coseno |
| UserKNN k=30 pearson | 30 | Pearson |

### 3.2 Filtrado colaborativo basado en ítems (ItemKNN)

Análogo a UserKNN, con las mismas combinaciones de k y similitud: 6 configuraciones.

| Configuración | k | Similitud |
|---|---|---|
| ItemKNN k=10 cosine | 10 | Coseno |
| ItemKNN k=10 pearson | 10 | Pearson |
| ItemKNN k=20 cosine | 20 | Coseno |
| ItemKNN k=20 pearson | 20 | Pearson |
| ItemKNN k=30 cosine | 30 | Coseno |
| ItemKNN k=30 pearson | 30 | Pearson |

### 3.3 Factorización matricial (SVD)

Descomposición en valores singulares con número de factores latentes k ∈ {5, 10, 20, 30}: 4 configuraciones.

| Configuración | Factores latentes |
|---|---|
| SVD f=5 | 5 |
| SVD f=10 | 10 |
| SVD f=20 | 20 |
| SVD f=30 | 30 |

### 3.4 Recomendador basado en contenido (CB-TF-IDF)

Construye perfiles de ítem mediante TF-IDF sobre las etiquetas asignadas por los usuarios (vocabulario de hasta 5.000 términos). El perfil de usuario se calcula como promedio ponderado por rating de los vectores TF-IDF de los ítems valorados en entrenamiento. La predicción r̂(u,i) se obtiene como media ponderada por similitud coseno de los ratings del usuario; si la similitud agregada es nula, se recurre a la proyección del perfil global del usuario sobre el vector del ítem.

---

## 4. Protocolo experimental

La evaluación utiliza las 5 particiones pre-proporcionadas de validación cruzada estratificada (80 % entrenamiento / 20 % test). Los resultados reportados son la media aritmética de las cinco particiones. El umbral de relevancia para las métricas de ranking se fija en 3,5. Los usuarios e ítems desconocidos en test se excluyen de la evaluación. La semilla aleatoria utilizada en todos los modelos es 1.

---

## 5. Métricas de evaluación

### 5.1 Error de predicción de rating

**MAE (Mean Absolute Error):** error medio absoluto entre rating predicho y real, $\text{MAE} = \frac{1}{|\mathcal{T}|}\sum_{(u,i)\in\mathcal{T}} |\hat{r}_{ui} - r_{ui}|$. Valores más bajos indican mayor precisión. Es la métrica más robusta frente a outliers entre las dos métricas de error.

**RMSE (Root Mean Squared Error):** raíz del error cuadrático medio, $\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u,i)\in\mathcal{T}} (\hat{r}_{ui} - r_{ui})^2}$. Penaliza más que MAE los errores grandes al elevar al cuadrado las diferencias, siendo más sensible a predicciones muy desviadas.

Para el recomendador basado en contenido (CB-TF-IDF), las métricas MAE y RMSE son orientativas, ya que el modelo no está optimizado para la predicción numérica de ratings sino para la similitud semántica entre ítems.

### 5.2 Métricas de ranking

**Precision\@N:** fracción de los N ítems recomendados que son relevantes (rating ≥ umbral). Mide la exactitud de la lista sin considerar el orden ni los ítems relevantes no recomendados.

**Recall\@N:** fracción de los ítems relevantes del usuario que aparecen entre los N recomendados. Mide la cobertura del sistema.

**F1\@N:** media armónica de Precision\@N y Recall\@N. Permite comparar configuraciones cuando precisión y recall evolucionan en sentidos contrarios al variar N o los parámetros del modelo.

**NDCG\@N (Normalized Discounted Cumulative Gain):** métrica sensible al orden que penaliza logarítmicamente los ítems relevantes situados en posiciones bajas. Normalizada entre 0 y 1 respecto al ranking ideal. Un NDCG alto indica que los ítems más relevantes ocupan las primeras posiciones de la lista.

---

## 6. Resultados

Los valores reportados son la media de las 5 particiones de validación cruzada.

### Tabla completa de resultados

| Modelo | MAE | RMSE | P@3 | R@3 | F1@3 | NDCG@3 | P@5 | R@5 | F1@5 | NDCG@5 | P@10 | R@10 | F1@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UserKNN k=10 cosine | 0.7180 | 0.8618 | 0.3024 | 0.0989 | 0.1427 | 0.3077 | 0.2895 | 0.1571 | 0.1941 | 0.3009 | 0.2659 | 0.2873 | 0.2628 | 0.3101 |
| UserKNN k=10 pearson | 0.6989 | 0.8448 | 0.3158 | 0.1045 | 0.1502 | 0.3214 | 0.3016 | 0.1655 | 0.2033 | 0.3139 | 0.2762 | 0.3008 | 0.2738 | 0.3236 |
| UserKNN k=20 cosine | 0.7010 | 0.8403 | 0.3303 | 0.1101 | 0.1578 | 0.3368 | 0.3125 | 0.1723 | 0.2112 | 0.3268 | 0.2807 | 0.3067 | 0.2790 | 0.3325 |
| UserKNN k=20 pearson | 0.6778 | 0.8187 | 0.3470 | 0.1169 | 0.1671 | 0.3553 | 0.3267 | 0.1822 | 0.2222 | 0.3436 | 0.2920 | 0.3220 | 0.2910 | 0.3491 |
| UserKNN k=30 cosine | 0.6941 | 0.8320 | 0.3458 | 0.1161 | 0.1661 | 0.3531 | 0.3262 | 0.1814 | 0.2216 | 0.3420 | 0.2889 | 0.3182 | 0.2879 | 0.3454 |
| UserKNN k=30 pearson | 0.6699 | 0.8094 | 0.3638 | 0.1242 | 0.1765 | 0.3735 | 0.3405 | 0.1919 | 0.2328 | 0.3598 | 0.3000 | 0.3332 | 0.2998 | 0.3625 |
| ItemKNN k=10 cosine | 0.7134 | 0.8785 | 0.3367 | 0.1149 | 0.1630 | 0.3469 | 0.3131 | 0.1762 | 0.2136 | 0.3327 | 0.2762 | 0.3074 | 0.2760 | 0.3349 |
| ItemKNN k=10 pearson | 0.6643 | 0.8069 | 0.3825 | 0.1323 | 0.1872 | 0.3972 | 0.3480 | 0.1984 | 0.2394 | 0.3751 | 0.2946 | 0.3311 | 0.2956 | 0.3680 |
| ItemKNN k=20 cosine | 0.7204 | 0.8777 | 0.3049 | 0.1040 | 0.1472 | 0.3128 | 0.2868 | 0.1614 | 0.1952 | 0.3030 | 0.2590 | 0.2878 | 0.2584 | 0.3101 |
| ItemKNN k=20 pearson | 0.6778 | 0.8175 | 0.3810 | 0.1322 | 0.1869 | 0.3963 | 0.3447 | 0.1969 | 0.2374 | 0.3729 | 0.2880 | 0.3243 | 0.2892 | 0.3629 |
| ItemKNN k=30 cosine | 0.7277 | 0.8813 | 0.2880 | 0.0994 | 0.1398 | 0.2951 | 0.2731 | 0.1554 | 0.1867 | 0.2878 | 0.2482 | 0.2778 | 0.2480 | 0.2971 |
| ItemKNN k=30 pearson | 0.6901 | 0.8300 | 0.3826 | 0.1331 | 0.1881 | 0.3992 | 0.3438 | 0.1968 | 0.2371 | 0.3739 | 0.2841 | 0.3206 | 0.2855 | 0.3613 |
| SVD f=5 | 0.6443 | 0.7869 | 0.4169 | 0.1488 | 0.2082 | 0.4304 | 0.3812 | 0.2244 | 0.2669 | 0.4094 | 0.3231 | 0.3748 | 0.3284 | 0.4064 |
| SVD f=10 | 0.6323 | 0.7738 | 0.4396 | 0.1584 | 0.2209 | 0.4543 | 0.4013 | 0.2376 | 0.2818 | 0.4318 | 0.3375 | 0.3920 | 0.3433 | 0.4268 |
| SVD f=20 | 0.6228 | 0.7638 | 0.4610 | 0.1674 | 0.2328 | 0.4765 | 0.4189 | 0.2486 | 0.2946 | 0.4516 | 0.3504 | 0.4077 | 0.3566 | 0.4453 |
| **SVD f=30** | **0.6195** | **0.7604** | **0.4701** | **0.1711** | **0.2377** | **0.4866** | **0.4267** | **0.2536** | **0.3002** | **0.4609** | **0.3557** | **0.4135** | **0.3618** | **0.4530** |
| CB-TF-IDF | 0.6923 | 0.8407 | 0.3523 | 0.1219 | 0.1722 | 0.3631 | 0.3268 | 0.1862 | 0.2245 | 0.3480 | 0.2880 | 0.3227 | 0.2884 | 0.3513 |

### 6.1 Error de predicción: MAE y RMSE

| Modelo | MAE | RMSE |
|---|---|---|
| **SVD f=30** | **0.6195** | **0.7604** |
| SVD f=20 | 0.6228 | 0.7638 |
| SVD f=10 | 0.6323 | 0.7738 |
| SVD f=5 | 0.6443 | 0.7869 |
| ItemKNN k=10 pearson | 0.6643 | 0.8069 |
| UserKNN k=30 pearson | 0.6699 | 0.8094 |
| UserKNN k=20 pearson | 0.6778 | 0.8187 |
| ItemKNN k=20 pearson | 0.6778 | 0.8175 |
| UserKNN k=30 cosine | 0.6941 | 0.8320 |
| ItemKNN k=30 pearson | 0.6901 | 0.8300 |
| CB-TF-IDF | 0.6923 | 0.8407 |
| UserKNN k=10 pearson | 0.6989 | 0.8448 |
| UserKNN k=20 cosine | 0.7010 | 0.8403 |
| ItemKNN k=10 cosine | 0.7134 | 0.8785 |
| UserKNN k=10 cosine | 0.7180 | 0.8618 |
| ItemKNN k=20 cosine | 0.7204 | 0.8777 |
| ItemKNN k=30 cosine | 0.7277 | 0.8813 |

Los modelos de factorización matricial (SVD) obtienen los menores valores de MAE y RMSE en todas sus configuraciones. SVD f=30 lidera con MAE = 0,6195 y RMSE = 0,7604. La diferencia MAE–RMSE oscila entre 0,14 y 0,17 en todos los modelos, lo que indica una distribución de errores moderadamente asimétrica sin outliers extremos dominantes.

Dentro de SVD, el error disminuye monótonamente al aumentar los factores latentes, aunque con rendimientos marginales decrecientes: el salto de f=5 a f=10 reduce el MAE en 0,012, mientras que el de f=20 a f=30 lo reduce solo en 0,003.

La similitud de Pearson supera sistemáticamente a la similitud coseno en ambas variantes KNN. La corrección de sesgo que aplica Pearson —normalizando las diferencias individuales de escala de valoración— es determinante en este conjunto de datos, donde los usuarios presentan patrones de valoración heterogéneos. En ItemKNN con similitud coseno se observa un comportamiento anómalo: el MAE empeora al aumentar k (de 0,7134 a 0,7277), lo que sugiere que los vecinos distantes sin corrección de sesgo contaminan las predicciones.

### 6.2 Calidad del ranking: Precision, Recall y F1

| Modelo | P@3 | R@3 | F1@3 | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 |
|---|---|---|---|---|---|---|---|---|---|
| **SVD f=30** | **0.4701** | **0.1711** | **0.2377** | **0.4267** | **0.2536** | **0.3002** | **0.3557** | **0.4135** | **0.3618** |
| SVD f=20 | 0.4610 | 0.1674 | 0.2328 | 0.4189 | 0.2486 | 0.2946 | 0.3504 | 0.4077 | 0.3566 |
| SVD f=10 | 0.4396 | 0.1584 | 0.2209 | 0.4013 | 0.2376 | 0.2818 | 0.3375 | 0.3920 | 0.3433 |
| SVD f=5 | 0.4169 | 0.1488 | 0.2082 | 0.3812 | 0.2244 | 0.2669 | 0.3231 | 0.3748 | 0.3284 |
| ItemKNN k=10 pearson | 0.3825 | 0.1323 | 0.1872 | 0.3480 | 0.1984 | 0.2394 | 0.2946 | 0.3311 | 0.2956 |
| UserKNN k=30 pearson | 0.3638 | 0.1242 | 0.1765 | 0.3405 | 0.1919 | 0.2328 | 0.3000 | 0.3332 | 0.2998 |
| ItemKNN k=30 pearson | 0.3826 | 0.1331 | 0.1881 | 0.3438 | 0.1968 | 0.2371 | 0.2841 | 0.3206 | 0.2855 |
| ItemKNN k=20 pearson | 0.3810 | 0.1322 | 0.1869 | 0.3447 | 0.1969 | 0.2374 | 0.2880 | 0.3243 | 0.2892 |
| CB-TF-IDF | 0.3523 | 0.1219 | 0.1722 | 0.3268 | 0.1862 | 0.2245 | 0.2880 | 0.3227 | 0.2884 |

Al aumentar N de 3 a 10, la precisión disminuye mientras el recall aumenta. La F1 tiende a crecer entre N=3 y N=10 porque la ganancia en recall domina: para SVD f=30, la F1 sube de 0,2377 a 0,3618. Este patrón es consistente en todos los modelos.

SVD supera a todos los métodos KNN en P, R y F1 para cualquier valor de N. SVD f=5 (F1\@10 = 0,3284) ya supera al mejor KNN (ItemKNN k=10 pearson, F1\@10 = 0,2956), lo que confirma que incluso la factorización con pocos factores captura relaciones latentes más útiles que la similitud directa entre vectores de rating.

Dentro de los KNN, Pearson supera a coseno en todos los escenarios. Para ItemKNN, la diferencia es especialmente marcada: ItemKNN k=10 pearson obtiene F1\@10 = 0,2956 frente a 0,2760 de su homólogo con coseno. La elección de la similitud resulta más determinante que el valor de k. Para ItemKNN con coseno, F1\@10 empeora monotónicamente al aumentar k (de 0,2760 a 0,2584 y 0,2480), mientras que con Pearson los resultados son más estables.

El recomendador CB-TF-IDF alcanza F1\@10 = 0,2884, equiparable a los mejores KNN-Pearson. Su principal diferencial es la capacidad de recomendar ítems sin historial de valoraciones (cold start de ítem), imposible para los métodos colaborativos.

### 6.3 Calidad del ordenamiento: NDCG

| Modelo | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|
| **SVD f=30** | **0.4866** | **0.4609** | **0.4530** |
| SVD f=20 | 0.4765 | 0.4516 | 0.4453 |
| SVD f=10 | 0.4543 | 0.4318 | 0.4268 |
| SVD f=5 | 0.4304 | 0.4094 | 0.4064 |
| ItemKNN k=10 pearson | 0.3972 | 0.3751 | 0.3680 |
| ItemKNN k=30 pearson | 0.3992 | 0.3739 | 0.3613 |
| ItemKNN k=20 pearson | 0.3963 | 0.3729 | 0.3629 |
| UserKNN k=30 pearson | 0.3735 | 0.3598 | 0.3625 |
| CB-TF-IDF | 0.3631 | 0.3480 | 0.3513 |

SVD f=30 lidera con NDCG\@10 = 0,4530. SVD f=5 (NDCG\@10 = 0,4064) supera ampliamente al mejor KNN (ItemKNN k=10 pearson, NDCG\@10 = 0,3680), confirmando la superioridad del modelo latente en la ordenación de ítems relevantes.

ItemKNN-Pearson es especialmente competitivo para listas cortas: ItemKNN k=10 pearson alcanza NDCG\@3 = 0,3972, mientras que para N=10 su ventaja sobre UserKNN se reduce (0,3680 vs 0,3625 del mejor UserKNN). Esto sugiere que ItemKNN-Pearson coloca los ítems más relevantes en las primeras posiciones con mayor eficacia que UserKNN, pero ambas familias convergen para listas largas.

Para ItemKNN coseno, el NDCG también empeora al aumentar k: NDCG\@10 cae de 0,3349 (k=10) a 0,3101 (k=20) y 0,2971 (k=30), alineándose con el patrón observado en F1.

---

## 7. Análisis comparativo

Los resultados muestran una jerarquía clara entre métodos: SVD domina en todas las métricas, seguido de los métodos KNN con Pearson, el recomendador de contenido y los KNN con coseno.

**SVD:** la factorización matricial supera a todos los métodos basados en similitud directa incluso con el menor número de factores (f=5). El aprendizaje de representaciones latentes en un espacio de dimensión reducida captura correlaciones de segundo orden entre usuarios e ítems que no son accesibles mediante similitudes de vecindad. El número óptimo de factores para este catálogo de 100 ítems se sitúa en 20–30, con rendimientos marginales decrecientes más allá de f=20.

**Similitud de Pearson frente a coseno:** en ambas familias KNN, Pearson supera sistemáticamente a coseno. La corrección del sesgo de escala personal es crítica cuando los usuarios presentan diferentes tendencias de valoración. La ausencia de esta corrección en coseno produce vecinos aparentemente similares que en realidad comparten sesgo, no preferencias.

**Efecto de k en KNN:** para UserKNN-Pearson, incrementar k mejora consistentemente todas las métricas (el mejor es k=30). Para ItemKNN-Pearson, el k óptimo es k=10; valores mayores introducen ítems vecinos menos relevantes que diluyen la señal. Para ambas variantes con coseno, k=30 es la peor configuración, lo que confirma que la similitud coseno sin corrección de sesgo no es escalable en número de vecinos.

**CB-TF-IDF:** obtiene resultados de ranking comparables a los mejores KNN (F1\@10 = 0,2884 frente a 0,2956 de ItemKNN k=10 pearson) sin utilizar ninguna información colaborativa. Su ventaja estratégica reside en la resolución del problema de cold start de ítem, al poder generar recomendaciones para películas sin historial de valoraciones.

**Elección según objetivo:**

| Objetivo | Configuración recomendada |
|---|---|
| Mínimo error de predicción (MAE/RMSE) | SVD f=30 |
| Máximo F1\@N y NDCG\@N | SVD f=30 |
| Equilibrio rendimiento–coste | SVD f=20 |
| Cold start de ítem nuevo | CB-TF-IDF |
| Interpretabilidad / vecinos explícitos | UserKNN k=30 pearson |

---

## 8. Conclusiones

La evaluación comparativa sobre cinco particiones de validación cruzada muestra que SVD con 20–30 factores latentes es la opción más robusta, liderando en MAE (0,6195), RMSE (0,7604), F1\@10 (0,3618) y NDCG\@10 (0,4530). La elección de la similitud de Pearson sobre coseno es el factor más determinante en los métodos KNN, con mayor impacto que el propio valor de k. El recomendador basado en contenido CB-TF-IDF, aunque inferior a SVD en todas las métricas cuantitativas, ofrece capacidad de recomendación en escenarios de cold start de ítem, complementando eficazmente a los métodos colaborativos en una arquitectura híbrida.
