# Image SEO SuperName

## Descripción

`image_seo_supername.py` es una herramienta de línea de comandos que ayuda a optimizar el SEO de tus imágenes mediante el renombramiento inteligente de archivos basado en el contexto proporcionado y análisis con inteligencia artificial. La herramienta genera nombres de archivo SEO-friendly utilizando información relevante sobre la marca, producto, categoría y palabras clave, además de aprovechar las capacidades de IA para la extracción de contexto y optimización de palabras clave.

## Características

- Generación de nombres de archivos optimizados para SEO
- **NOVEDAD**: Análisis de imágenes con IA usando la API de Mistral (pixtral-12b-2409)
- **NOVEDAD**: Generación de texto alternativo (alt text) para mejorar SEO
- **NOVEDAD**: Extracción de palabras clave basada en el contenido visual
- **NOVEDAD**: Sistema de logging detallado para un mejor seguimiento y solución de problemas
- Soporte multilingüe (Español e Inglés)
- Renombramiento automático con numeración secuencial
- Registro del historial de renombramientos
- Validación de nombres SEO-friendly
- Normalización de texto (eliminación de acentos, caracteres especiales)
- Filtrado de palabras vacías (stop words)
- Modo seguro para preservar archivos originales
- Funcionalidad de restauración para recuperar nombres originales
- Verificación de duplicados antes del procesamiento
- **NOVEDAD**: Campos adicionales para ubicación geográfica y contexto

## Requisitos

- Python 3.6 o superior
- Bibliotecas: pathlib, tqdm, dotenv, **requests**, **Pillow**, **tenacity**
- **NOVEDAD**: Clave de API de Mistral (para funciones de IA)

## Instalación

1. Clona el repositorio o descarga el script `image_seo_supername.py`.
2. Instala las dependencias:

```bash
pip install python-dotenv tqdm requests Pillow tenacity
```

3. **NOVEDAD**: Para funciones de IA, obtén una clave API de Mistral y configúrala:

```bash
# Opción 1: Como variable de entorno
export MISTRAL_API_KEY="tu-clave-api"

# Opción 2: Crear archivo .env
echo "MISTRAL_API_KEY=tu-clave-api" > .env

# Opción 3: Proporcionar directamente en el comando
python image_seo_supername.py --rename --input /ruta/imagenes --ai --api-key "tu-clave-api"
```

## Uso básico

El script tiene tres modos distintos de operación:

### 1. Modo Renombrar (predeterminado)

```bash
# Renombrado estándar (sin IA)
python image_seo_supername.py --rename --input /ruta/a/imagenes

# Con análisis potenciado por IA
python image_seo_supername.py --rename --input /ruta/a/imagenes --ai
```

Esto procesará todas las imágenes en el directorio especificado y creará copias con nombres SEO-optimizados.

### 2. Modo Restaurar

```bash
python image_seo_supername.py --restore --history /ruta/a/rename_history.json
```

Restaura los archivos a sus nombres originales utilizando el historial de renombramiento.

### 3. Mostrar Opciones de Recuperación

```bash
python image_seo_supername.py --recovery-options --history /ruta/a/rename_history.json
```

Muestra las opciones disponibles para recuperar archivos basándose en el historial.

## Parámetros

| Parámetro | Abreviación | Descripción | Requerido | Valor predeterminado |
|-----------|-------------|-------------|-----------|----------------------|
| `--rename` | `-r` | Activa el modo de renombrado | Sí (uno de los modos) | - |
| `--restore` | `-s` | Activa el modo de restauración | Sí (uno de los modos) | - |
| `--recovery-options` | `-o` | Muestra opciones de recuperación | Sí (uno de los modos) | - |
| `--input` | `-i` | Directorio de entrada con las imágenes | Sí (en modo renombrar) | - |
| `--output` | `-O` | Directorio de salida | No | Igual que el directorio de entrada |
| `--language` | `-l` | Idioma para los prompts (en=Inglés, es=Español) | No | `es` |
| `--move` | `-m` | Usar modo MOVE en lugar de COPY (PELIGROSO) | No | `False` |
| `--history` | `-H` | Ruta al archivo de historial | Sí (en modos restaurar/recuperación) | - |
| `--force` | `-f` | Sobrescribir archivos existentes durante la restauración | No | `False` |
| `--ai` | `-a` | Habilitar análisis de imágenes con IA | No | `False` |
| `--api-key` | `-k` | Clave API de Mistral para funciones de IA | No | Desde variable de entorno |
| `--verbose` | `-v` | Activar registro detallado de operaciones | No | `False` |
| `--log-file` | - | Ruta al archivo de log | No | `image_seo_supername.log` |
| `--no-log-file` | - | Deshabilitar registro en archivo | No | `False` |

## Ejemplos de uso

### Modo Renombrar (Seguro - preserva originales)

```bash
python image_seo_supername.py --rename --input "./mis_imagenes"
```

### Renombrar con Análisis de IA

```bash
python image_seo_supername.py --rename --input "./mis_imagenes" --ai
```

### Modo Renombrar (Peligroso - modifica originales)

```bash
python image_seo_supername.py --rename --input "./mis_imagenes" --move
```

### Especificar directorio de salida diferente

```bash
python image_seo_supername.py --rename --input "./imagenes_originales" --output "./imagenes_optimizadas"
```

### Cambiar el idioma a inglés

```bash
python image_seo_supername.py --rename --input "./mis_imagenes" --language en
```

### Restaurar archivos

```bash
python image_seo_supername.py --restore --history "./rename_history.json"
```

### Restaurar archivos forzando sobrescritura

```bash
python image_seo_supername.py --restore --history "./rename_history.json" --force
```

### Ver opciones de recuperación

```bash
python image_seo_supername.py --recovery-options --history "./rename_history.json"
```

### Uso con opciones de logging

```bash
# Habilitar logging detallado
python image_seo_supername.py --rename --input "./mis_imagenes" --verbose

# Especificar una ruta personalizada para el archivo de log
python image_seo_supername.py --rename --input "./mis_imagenes" --log-file "./logs/renombrado_20231115.log"

# Deshabilitar el registro en archivo (solo consola)
python image_seo_supername.py --rename --input "./mis_imagenes" --no-log-file

# Combinación con otras opciones
python image_seo_supername.py --rename --input "./mis_imagenes" --ai --verbose --log-file "./logs/renombrado_con_ia.log"
```

## Proceso de renombramiento

Durante la ejecución, el script te solicitará:

1. **Marca/Empresa**: El nombre de la marca o empresa (obligatorio).
2. **Producto/Servicio**: El nombre del producto o servicio (obligatorio).
3. **Categoría**: La categoría del producto o servicio (obligatorio).
4. **Ubicación geográfica**: Región o área relevante (opcional).
5. **Palabras clave**: Términos relacionados separados por comas (opcional).
6. **Contexto adicional**: Información extra para mejorar el análisis de IA (cuando IA está habilitada).

Con esta información, el programa:

1. Normaliza y limpia todos los textos (elimina acentos, convierte a minúsculas, etc.).
2. Elimina palabras vacías (stop words) que no aportan valor SEO.
3. Si la IA está habilitada, analiza cada imagen para extraer contenido visual relevante.
4. Construye nombres de archivo usando componentes relevantes de la entrada del usuario y análisis de IA.
5. Añade numeración secuencial para evitar duplicados.
6. Limita la longitud del nombre a 60 caracteres para cumplir con las mejores prácticas SEO.
7. Renombra los archivos y guarda un registro de los cambios en `rename_history.json`.
8. Almacena el texto alternativo generado por IA para cada imagen.

## Sistema de Logging

El script implementa un sistema de logging detallado para facilitar el seguimiento y la solución de problemas:

1. **Niveles de log**:
   - **Normal**: Registra información básica sobre operaciones (INFO)
   - **Detallado**: Con `--verbose`, registra información detallada de depuración (DEBUG)

2. **Destinos de log**:
   - **Consola**: Muestra mensajes de nivel INFO o superior
   - **Archivo**: Registra todos los mensajes (incluyendo DEBUG con `--verbose`)

3. **Información registrada**:
   - Fecha y hora de cada operación
   - Nivel de severidad (INFO, WARNING, ERROR, etc.)
   - En modo detallado: nombre del archivo y número de línea
   - Mensajes descriptivos de cada operación
   - Errores y excepciones con trazas completas

4. **Opciones de configuración**:
   - `--verbose`: Aumenta el nivel de detalle
   - `--log-file`: Especifica una ruta personalizada para el archivo de log
   - `--no-log-file`: Deshabilita el registro en archivo

5. **Archivo de log**:
   - Formato: texto plano con entradas separadas por línea
   - Ubicación predeterminada: `image_seo_supername.log` en el directorio actual
   - Modo de apertura: append (añade a un archivo existente)
   - Codificación: UTF-8 para soporte completo de caracteres

El sistema de logging ayuda a:
- Identificar problemas durante el procesamiento
- Mantener un registro histórico de operaciones
- Facilitar la depuración en caso de fallos
- Proporcionar información detallada para soporte técnico

## Estructura de nombres generados

### Nomenclatura Estándar (sin IA)
Los nombres de archivo generados siguen este patrón:

```
[palabra-clave1]-[palabra-clave2]-[producto]-[marca]-[categoria]-[###].[extension]
```

### Nomenclatura Mejorada con IA
Cuando el análisis de IA está activado, los nombres siguen este patrón mejorado:

```
[marca]-[palabra_clave_ia]-[producto]-[ubicacion]-[categoria]-[###].[extension]
```

Donde:
- La marca siempre tiene prioridad (mayor valor SEO)
- Las palabras clave extraídas por IA proporcionan contexto más rico sobre el contenido de la imagen
- Los componentes se separan con guiones
- Se eliminan espacios y caracteres especiales
- Se incluye número secuencial de tres dígitos al final
- La longitud total se limita a 60 caracteres

## En caso de emergencia

Si has perdido archivos debido a un error en el script:

1. **NO ELIMINES** el archivo `rename_history.json`
2. **REVISA** el archivo de log para identificar el problema
3. Ejecuta `python image_seo_supername.py --recovery-options --history "ruta/al/rename_history.json"` 
4. Sigue las instrucciones proporcionadas para recuperar tus archivos.