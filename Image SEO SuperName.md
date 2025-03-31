# Image SEO SuperName

## Descripción

`image_seo_supername.py` es una herramienta de línea de comandos que ayuda a optimizar el SEO de tus imágenes mediante el renombramiento inteligente de archivos basado en el contexto proporcionado y análisis con inteligencia artificial. La herramienta genera nombres de archivo SEO-friendly utilizando información relevante sobre la marca, producto, categoría y palabras clave, además de aprovechar las capacidades de IA para la extracción de contexto y optimización de palabras clave.

## Características

- Generación de nombres de archivos optimizados para SEO
- **NOVEDAD**: Análisis de imágenes con IA usando la API de Mistral (pixtral-12b-2409)
- **NOVEDAD**: Generación de texto alternativo (alt text) para mejorar SEO
- **NOVEDAD**: Extracción de palabras clave basada en el contenido visual
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
| **NUEVO:** `--ai` | `-a` | Habilitar análisis de imágenes con IA | No | `False` |
| **NUEVO:** `--api-key` | `-k` | Clave API de Mistral para funciones de IA | No | Desde variable de entorno |

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

## Proceso de renombramiento

Durante la ejecución, el script te solicitará:

1. **Marca/Empresa**: El nombre de la marca o empresa (obligatorio).
2. **Producto/Servicio**: El nombre del producto o servicio (obligatorio).
3. **Categoría**: La categoría del producto o servicio (obligatorio).
4. **NUEVO:** **Ubicación geográfica**: Región o área relevante (opcional).
5. **Palabras clave**: Términos relacionados separados por comas (opcional).
6. **NUEVO:** **Contexto adicional**: Información extra para mejorar el análisis de IA (cuando IA está habilitada).

Con esta información, el programa:

1. Normaliza y limpia todos los textos (elimina acentos, convierte a minúsculas, etc.).
2. Elimina palabras vacías (stop words) que no aportan valor SEO.
3. **NUEVO:** Si la IA está habilitada, analiza cada imagen para extraer contenido visual relevante.
4. Construye nombres de archivo usando componentes relevantes de la entrada del usuario y análisis de IA.
5. Añade numeración secuencial para evitar duplicados.
6. Limita la longitud del nombre a 60 caracteres para cumplir con las mejores prácticas SEO.
7. Renombra los archivos y guarda un registro de los cambios en `rename_history.json`.
8. **NUEVO:** Almacena el texto alternativo generado por IA para cada imagen.

## Estructura de nombres generados

### Nomenclatura Estándar (sin IA)
Los nombres de archivo generados siguen este patrón:

```
[palabra-clave1]-[palabra-clave2]-[producto]-[marca]-[categoria]-[###].[extension]
```

### **NUEVO:** Nomenclatura Mejorada con IA
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

## **NUEVO:** Proceso de Análisis con IA

Cuando el modo de IA está habilitado, ocurre lo siguiente para cada imagen:

1. La imagen se analiza utilizando el modelo pixtral-12b-2409 de Mistral
2. La IA extrae:
   - Palabras clave relevantes del contenido visual
   - Sujeto principal o producto visible
   - Características visuales (colores, estilo, composición)
   - Contexto/escenario de la imagen
   - Texto alternativo sugerido
3. Esta información se combina con el contexto proporcionado por el usuario para una nomenclatura óptima
4. Las sugerencias de texto alternativo se guardan en el archivo de historial

## Archivo de historial

El script crea un archivo `rename_history.json` en el directorio de salida que contiene un registro de todos los renombramientos realizados. Este archivo es fundamental para:

1. Rastrear todos los cambios realizados
2. Permitir la restauración de nombres originales
3. Facilitar recuperación en caso de errores
4. **NUEVO:** Almacenar el texto alternativo generado por IA para cada imagen

## Seguridad y Prevención de Pérdida de Datos

El script implementa varias medidas de seguridad:

- **Modo COPY por defecto**: Las imágenes originales se preservan por defecto
- **Confirmación explícita**: Se requiere confirmación para usar el modo MOVE
- **Verificación previa**: Se verifican duplicados antes del procesamiento
- **Historial preservado**: Se mantiene registro de todos los cambios
- **Guardado progresivo**: El historial se guarda periódicamente durante el procesamiento
- **Manejo de interrupciones**: Se guarda el historial si el script se interrumpe con Ctrl+C
- **NUEVO:** **Reintentos automáticos**: Para fallos de conexión con la API de Mistral

## Recuperación de Errores

Si ocurren problemas durante el renombramiento, el script proporciona las siguientes opciones de recuperación:

1. **Comando de restauración**: `--restore` para volver a los nombres originales
2. **Asistente de recuperación**: `--recovery-options` para ver opciones disponibles
3. **Restauración forzada**: `--force` para sobrescribir archivos existentes si es necesario
4. **NUEVO:** **Fallback a modo estándar**: Si falla el análisis de IA, usa el nombramiento estándar

## Mejores prácticas

- Usa palabras clave específicas y relevantes.
- Evita nombres demasiado genéricos.
- Incluye términos de búsqueda populares relacionados con tu contenido.
- Sé consistente con la nomenclatura de archivos.
- Utiliza categorías precisas.
- **IMPORTANTE**: Siempre haz una copia de seguridad antes de usar el modo MOVE.
- Verifica el resultado del renombramiento antes de eliminar archivos originales.
- **NUEVO:** Proporciona contexto detallado para mejorar el análisis de IA.
- **NUEVO:** Revisa los textos alternativos generados por la IA para optimización adicional.

## Solución de problemas

- **No se encuentran imágenes**: Verifica que la ruta de entrada sea correcta y contenga archivos de imagen compatibles.
- **Error de permisos**: Asegúrate de tener permisos de escritura en los directorios de entrada/salida.
- **Nombres no deseados**: Revisa los parámetros proporcionados y ajusta las palabras clave según sea necesario.
- **Archivos perdidos**: Utiliza la funcionalidad de restauración con el archivo de historial.
- **Error de sobreescritura**: Usa un directorio de salida diferente o comprueba por nombres duplicados.
- **NUEVO:** **Fallos en análisis de IA**: Verifica tu clave API y conexión a internet. El script usará nomenclatura estándar si el análisis falla.

## Limitaciones

- Procesa archivos con extensiones .jpg, .jpeg, .png, .gif y .webp.
- **NUEVO:** La API de Mistral tiene un límite de tamaño de archivo de 10MB para cargas de imágenes.
- **NUEVO:** Las funciones de IA requieren conexión a internet activa.
- **NUEVO:** Pueden aplicarse límites de tasa de la API según tu suscripción a Mistral.

## En caso de emergencia

Si has perdido archivos debido a un error en el script:

1. **NO ELIMINES** el archivo `rename_history.json`
2. Ejecuta `python image_seo_supername.py --recovery-options --history "ruta/al/rename_history.json"` 
3. Sigue las instrucciones proporcionadas para recuperar tus archivos.