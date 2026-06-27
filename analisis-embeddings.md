# Análisis y Explotación de Embeddings — BoxData

## La forma definitiva de recomendar (kernel regression + media geométrica)

### Intuición (sin tecnicismos)

**Hoy**: Agrupamos las pelis que ha visto un usuario en 2-4 grupos (acción, drama, terror...). Calculamos el punto medio de cada grupo (centroide). Para recomendar, comparamos cada candidata con esos puntos medios.

**La nueva idea**: En vez de comprimir todo el historial en 4 puntos, usamos **todas las pelis que ha visto** para crear una "función de gusto" continua. Esta función, dado el embedding de cualquier peli, predice qué nota le pondría el usuario.

Es como tener un mapa en 768 dimensiones donde cada peli que ha visto es una montaña con altura = su nota. Entre montañas, el terreno se interpola suavemente. El mapa completo es su función de gusto.

### Matemáticamente (un poco)

**Regresión por kernel (Nadaraya-Watson)** — el valor de la función en cualquier punto se calcula así:

```
f(embedding_candidato) = suma(peso_i * nota_i) / suma(peso_i)

donde peso_i = exp(-distancia² / sigma²)
y distancia = 1 - similitud_coseno(embedding_candidato, embedding_peli_vista_i)
```

Cada peli del historial contribuye, pero las más parecidas pesan mucho más. `sigma` controla lo "suave" que es: sigma grande = suave (casi una media global), sigma pequeño = más fiel a cada punto.

**Ventajas sobre los centroides:**
- No pierdes información comprimiendo en K grupos
- Es suave: dos pelis parecidas tienen notas parecidas
- Preserva la complejidad del usuario
- Tiene derivada (se puede calcular hacia dónde sube)

### Consenso para grupo: media geométrica

Dados N usuarios con sus funciones f_alice, f_bob, f_carlos..., la función del grupo es:

```
G(x) = (f_alice(x) * f_bob(x) * f_carlos(x)) ^ (1/N)
```

**Por qué es mejor que las alternativas:**
- **Min** es muy conservadora: una persona con gustos raros veta todo
- **Media aritmética** diluye: una peli que Alice odia (0.1) y Bob ama (0.9) da 0.5
- **Media geométrica**: si Alice da 0.1 y Bob 0.9 → sqrt(0.1*0.9) = 0.3 (penaliza pero no veta)
- Si ambos dan 0.9 → 0.9. Si ambos dan 0.5 → 0.5
- Naturalmente premia el consenso sin ser demasiado agresiva

### Búsqueda de candidatos: gradient ascent multi-start

En vez de pedir candidatos a ChromaDB y luego puntuarlos, exploramos el espacio directamente buscando los máximos de G:

1. **Puntos de inicio**: la query del usuario, centroides actuales, 10-20 puntos aleatorios
2. **Gradient ascent**: desde cada punto, miramos en qué dirección sube G y nos movemos
3. **Repetimos** hasta llegar a un pico
4. **Recolectamos picos** distintos (máximos locales)
5. **Buscamos la peli real más cercana** a cada pico en ChromaDB

### Optimizaciones para que sea rápido

**Problema**: evaluar f(embedding) para un usuario requiere comparar contra todo su historial (ej. 200 pelis). Si hacemos gradient ascent con 100 pasos desde 20 puntos de inicio, con 3 usuarios, son 20*100*3*200 = 1.2M de cosenos por consulta.

**Soluciones:**

1. **Ignorar embeddings lejanos**: pelis con distancia > 0.3 tienen peso prácticamente cero (exp(-0.3²/sigma²) es casi 0). Ni las calculas. Con embeddings normalizados, el coseno se puede estimar rápido con productos parciales y cortar cuando sea evidente que la distancia es grande.

2. **Precalcular índices**: ChromaDB ya tiene ANN (approximate nearest neighbors). En vez de comparar contra las 200, buscas las 20 más cercanas y solo usas esas. El resto tienen peso ~0. Básicamente k-NN pero con kernel en vez de media simple.

3. **No hacer gradient ascent sobre ChromaDB**: Usa los centroides actuales + query + puntos aleatorios como candidatos, evalúa G en ellos, y refina con gradient ascent SOLO en los alrededores de los mejores. No exploras todo el espacio de 768 dimensiones, solo las regiones prometedoras.

4. **Batch**: si tienes 100 candidatas, evalúas G para todas a la vez usando operaciones vectoriales (numpy). 100 candidatas * 200 historial * 768 dimensiones = 15M de multiplicaciones. Numpy las hace en milisegundos.

### Coste real estimado

| Escenario | Tiempo aprox |
|-----------|-------------|
| 3 usuarios, 200 pelis c/u, 1000 candidatas (batch numpy) | ~0.1s |
| Lo mismo + gradient ascent 50 pasos desde 10 puntos | ~0.5s |
| 10 usuarios, 500 pelis c/u, mismo proceso | ~2-3s |
| Todo con ANN en ChromaDB (sin gradient ascent) | ~0.2s |

Es perfectamente manejable para una demo web. El cuello de botella real es ChromaDB, no el kernel.

### Cómo integrarlo con ChromaDB (híbrido práctico)

```
1. Para cada usuario: cargar historial (embeddings + ratings) en memoria
2. Query a ChromaDB con centroides actuales → top 200 candidatas
3. Para cada candidata:
   a. f_alice(x) = kernel_regression(x, historial_alice) → nota predicha
   b. f_bob(x) = kernel_regression(x, historial_bob) → nota predicha
   c. G(x) = (f_alice * f_bob) ^ 1/2 → consenso
4. Reordenar por G(x) → top 10
5. Opcional: gradient ascent desde los mejores puntos para refinar
```

No tiras ChromaDB, usas ChromaDB para el pool inicial y el kernel para el scoring fino. Lo mejor de ambos.

### 6. Gestión de sigma (el parámetro clave)

sigma controla cuánto se "suaviza" la función de gusto:

- **sigma muy pequeño** (~0.05): la función solo da notas altas para pelis casi idénticas a una del historial. Muy conservadora, no generaliza.
- **sigma grande** (~0.5): la función suaviza todo, pierde matices, se acerca a la media global del usuario.
- **sigma óptimo**: se calibra con validación cruzada — se ocultan algunos ratings conocidos, se predicen con distintos sigma, se mide el error.

Para empezar: sigma = 0.2 o 0.3 funciona bien en espacios de embeddings normalizados.

### Resumen del cambio

| Componente | Antes | Después |
|-----------|-------|---------|
| Representación usuario | 2-4 centroides (k-means) | Kernel regression sobre historial completo |
| Score por usuario | max cosine_sim(candidato, centroides) | f(candidato) via Nadaraya-Watson |
| Consenso grupo | mean - 0.5*std | Media geométrica de las funciones |
| Búsqueda candidatos | Query ChromaDB con media de centroides | ChromaDB + gradient ascent multi-start |
| Parámetros | K fijo, sin calibrar | sigma calibrable por validación cruzada |

---

## Otras formas de explotar los embeddings

### 🎯 Nivel 1: Análisis del espacio

#### 1. Visualizar el espacio con UMAP + HDBSCAN

Reducir 768 dimensiones a 2 con UMAP, clusterizar con HDBSCAN, dibujar un mapa interactivo.

**Qué descubrir:**
- ¿Los géneros de TMDB corresponden a regiones del espacio?
- ¿Aparecen grupos no etiquetados? (thriller conspirativo, horror psicológico japonés)
- ¿Hay pelis raras (outliers) o puente entre clusters?

**Requisito técnico (claude):** HDBSCAN sobre 768 dimensiones da resultados pobres por la maldición de la dimensionalidad. Reducir primero con UMAP a 10-20 dimensiones, luego HDBSCAN.

#### 2. Ejes de género (direcciones en el espacio)

```
eje_terror_comedia = centro_terror - centro_comedia
```

Cualquier película se proyecta en este eje. Asignación suave de géneros: 30% terror, 50% drama, 20% comedia.

Ejes posibles: Terror ↔ Comedia, Antiguo ↔ Moderno, Blockbuster ↔ Cine de autor, Occidental ↔ Oriental.

#### 3. Análisis temporal por décadas

Centroide por década desde 1900. ¿Cómo se mueve el cine? ¿Rupturas (60s, 70s)? ¿Diversificación u homogeneización?

#### 4. Analogías vectoriales

```
Alien - terror + comedia = ?
```

Buscar la peli real más cercana al vector resultante.

#### 5. Outliers y rareza

Distancia media a los 10 vecinos más cercanos. Pelis raras (alta distancia) vs típicas (baja distancia).

#### 6. Validar qué capturan los embeddings

¿Directores similares → embeddings cercanos? ¿Misma década? ¿Misma puntuación? Si no, revela qué está priorizando el modelo.

### 🎯 Nivel 2: Mejoras a los embeddings

#### 7. Richer enriched text

Añadir al texto del embedding: reviews populares de Letterboxd, palabras clave de TMDB, presupuesto/taquilla, país de origen, idiomas. Mejores embeddings → todo lo demás mejora.

### 🎯 Nivel 3: Arquitectura (ideas adicionales)

#### 8. Descomposición en arquetipos del catálogo

Clusterizar todo el catálogo con UMAP + HDBSCAN → 20-30 arquetipos fijos e interpretables ("acción palomitera", "drama de autor europeo"). Cada usuario es una mezcla de estos arquetipos según las pelis que ha visto. No requiere clusterizar por usuario.

#### 9. Attention (para cuando haya 100+ perfiles)

Un transformer pequeño que aprende dinámicamente qué pelis del historial de un usuario son relevantes para predecir su nota a una candidata específica. Para Blade Runner, mira más sus sci-fi vistas; para Amélie, más sus comedias románticas.

### 🎨 Nivel 4: Demo

#### 10. Mapa interactivo (Plotly)

Scatter UMAP 2D con hover para título, año, géneros. Demo visual.

#### 11. "Why this recommendation"

Mostrar por qué salió cada peli: "60% match con tu lado Nolan, 30% con tu lado Ghibli".

---

## Roadmap de implementación

### Fase 1: Nuevo motor de recomendación (kernel + media geométrica)
[completado]

### Fase 2: Calibrar sigma con validación cruzada
[completado]

- Para cada usuario, validación cruzada k-fold sobre su historial
- Prueba sigma candidates: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
- Elige el sigma con menor MAE
- Fallback a 0.3 si < 10 pelis
- CLI: `--auto-sigma` (activa calibración por usuario)

### Fase 3: Análisis del espacio (UMAP + HDBSCAN)
[completado]

- Reduce embeddings 768d → 10d (UMAP), clusteriza con HDBSCAN
- Reduce a 2d para visualización interactiva (Plotly)
- 14 clusters encontrados en 8481 películas
- Gráfico guardado en `output/embedding_map.html`
- Comando: `python main.py explore`

### Fase 4: Ejes de género
[completado]

- Calcula centroide por género (promedio de embeddings de pelis de ese género)
- Construye ejes semánticos: horror→comedy, action→drama, sci-fi→fantasy, thriller→romance
- Proyecta cada película en cada eje (top/bottom 5)
- Visualización de centroides en `output/genre_centroids.html`

### Fase 5: Mejorar enriched_text
[completado]

- `enriched_text()` ahora incluye `origin_countries` y `spoken_languages`
- Metadata de ChromaDB normalizada (sin None)
- Re-embedding completo de 8481 películas (~20s)

### Fase 6: Demo web
[completado]

- FastAPI en `web/app.py`
- Formulario: usuarios, query, género, diversity, auto-sigma
- Tabla de resultados con links a Letterboxd
- Sirve el mapa UMAP desde `/static/embedding_map.html`
- Arrancar: `python web/app.py` (puerto 8080)
