# Copilot Instructions — SLM

## 1. Objetivo

Este repositorio se usa para aprender y experimentar con modelos de lenguaje pequeños y con modelos tipo GPT desde cero.

El flujo de trabajo separa cuatro funciones:

- **VS Code + WSL**: desarrollo principal.
- **GitHub**: código, notebooks propios y configuración.
- **Google Colab**: ejecución remota, sobre todo cuando hace falta GPU.
- **Google Drive**: datos de entrada, checkpoints, modelos, resultados y otros artefactos persistentes.

Colab debe tratarse como un entorno de cálculo temporal, no como el entorno principal de desarrollo.

## 2. Estructura actual

```text
SLM/
├── .github/
│   └── instructions/
│       └── python.instructions.md
├── notebooks/
│   ├── book/
│   │   └── Chapter 2:_build_GPT_from_scratch.ipynb
│   └── mios/
│       └── My_Chapter 2:simple_bigram_model.ipynb
├── src/
│   └── slm/
│       ├── __init__.py
│       ├── ch2_simple_bigram.py
│       └── ch2_simple_gpt.py
└── .gitignore
```

### Regla especial: `notebooks/book/`

`notebooks/book/` contiene código procedente de un libro usado como material de referencia.

- No forma parte del código propio.
- No debe subirse al repositorio.
- Está incluida en `.gitignore`.
- No debe modificarse salvo petición expresa.
- Puede consultarse localmente como referencia.
- No copiar grandes bloques del libro al código propio; reimplementar los conceptos.

## 3. Regla principal

El código reutilizable debe vivir en `src/slm/`.

Los notebooks deben contener sobre todo:

- configuración;
- carga de datos;
- llamadas a funciones o clases de `slm`;
- experimentos;
- gráficos;
- resultados;
- explicaciones.

Evitar lógica extensa dentro de notebooks.

Ejemplo preferido:

```python
from slm.ch2_simple_gpt import BigramLanguageModel, train_model

model = BigramLanguageModel(...)
result = train_model(model, ...)
```

## 4. Desarrollo normal: VS Code + WSL

```bash
cd ~/projects/SLM
code .
```

Trabajar principalmente en:

```text
src/slm/
notebooks/mios/
tests/
```

Hacer en WSL, siempre que no haga falta GPU:

- edición;
- depuración;
- refactorización;
- pruebas;
- `mypy`;
- `ruff`;
- Git;
- experimentos pequeños en CPU.

## 5. GitHub

GitHub es la fuente del código entre WSL y Colab.

Flujo normal:

```bash
git status
git add .
git commit -m "Descripción breve del cambio"
git push
```

Guardar en GitHub:

```text
src/
notebooks/mios/
tests/
pyproject.toml
README.md
documentación
configuración
```

No guardar grandes datasets, pesos o resultados binarios en Git salvo causa clara.

Antes de cada commit comprobar que `notebooks/book/` sigue fuera del repositorio.

## 6. Google Drive

Google Drive es el almacenamiento persistente para los datos que usan los notebooks y para los resultados que generan.

Estructura recomendada:

```text
MyDrive/
└── SLM/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── checkpoints/
    ├── models/
    ├── results/
    └── logs/
```

Regla:

```text
Código                         → GitHub
Datos persistentes             → Google Drive
Resultados persistentes        → Google Drive
Trabajo temporal de Colab      → /content
```

No usar Google Drive para sincronizar el repositorio.

## 7. Montar Drive en Colab

```python
from google.colab import drive

drive.mount("/content/drive")
```

Definir rutas:

```python
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/SLM")

DATA_DIR = DRIVE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = DRIVE_ROOT / "checkpoints"
MODELS_DIR = DRIVE_ROOT / "models"
RESULTS_DIR = DRIVE_ROOT / "results"
LOGS_DIR = DRIVE_ROOT / "logs"
```

Crear carpetas si hace falta:

```python
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
```

## 8. Datos y resultados

### Datos de entrada

Los datasets persistentes deben vivir en Google Drive.

Ejemplo:

```text
/content/drive/MyDrive/SLM/data/raw/input.txt
```

### Resultados

Guardar en Drive lo que deba sobrevivir al cierre del runtime:

- checkpoints;
- pesos;
- métricas;
- CSV;
- JSON;
- figuras;
- logs;
- resultados de evaluación;
- artefactos de inferencia.

Ejemplo:

```python
torch.save(
    model.state_dict(),
    CHECKPOINTS_DIR / "model.pt",
)
```

### Trabajo temporal

Usar `/content` para:

```text
/content/input.txt
/content/cache/
/content/tmp/
```

El contenido de `/content` desaparece al reiniciar el runtime.

## 9. Rendimiento con Drive

No usar Google Drive como disco de trabajo intensivo si no hace falta.

Patrón recomendado:

1. leer o copiar datos desde Drive;
2. trabajar en `/content`;
3. guardar en Drive sólo lo que deba persistir.

Ejemplo:

```python
from shutil import copy2
from pathlib import Path

source = RAW_DATA_DIR / "input.txt"
local = Path("/content/input.txt")

copy2(source, local)
```

Esto importa sobre todo con:

- muchos ficheros pequeños;
- entrenamiento;
- checkpoints frecuentes;
- cargas repetidas;
- datasets medianos o grandes.

## 10. Colab

Los notebooks de `notebooks/mios/` pueden ejecutarse en Colab cuando interese usar GPU.

El runtime:

- es temporal;
- no comparte el sistema de archivos de WSL;
- no ve por sí solo `~/projects/SLM`;
- pierde `/content` al reiniciarse.

Por eso el código debe llegar a Colab desde GitHub.

## 11. Preparar un runtime nuevo de Colab

```python
!git clone https://github.com/<USUARIO>/<REPOSITORIO>.git
%cd <REPOSITORIO>
!pip install -e .
```

Después:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Si el repositorio ya está clonado:

```python
%cd /content/<REPOSITORIO>
!git pull
```

No volver a clonarlo.

## 12. VS Code y extensión de Colab

Un notebook puede abrirse en VS Code y usar un runtime de Colab como kernel.

Pero:

```text
filesystem WSL != filesystem Colab
```

Seleccionar un kernel de Colab no hace visible automáticamente `~/projects/SLM/src/`.

El código debe estar también en el runtime remoto mediante `git clone` o `git pull`.

## 13. Patrón recomendado para notebooks

1. Preparar el runtime si es nuevo.
2. Montar Drive.
3. Definir rutas.
4. Importar `slm`.
5. Definir configuración.
6. Cargar datos.
7. Copiar a `/content` lo que se vaya a usar de forma intensiva.
8. Instanciar el modelo.
9. Entrenar.
10. Evaluar.
11. Generar o inferir.
12. Guardar resultados persistentes en Drive.

Ejemplo de configuración:

```python
SEED = 42
BATCH_SIZE = ...
LEARNING_RATE = ...
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 14. Qué debe pasar del notebook a `src/slm`

Mover a `.py` cuando el código:

- se usa más de una vez;
- define una clase;
- define la arquitectura del modelo;
- implementa entrenamiento;
- implementa evaluación;
- contiene lógica reutilizable de datos;
- ocupa varias celdas;
- necesita pruebas;
- necesita tipos;
- interesa depurarlo desde VS Code.

Mantener en notebook cuando:

- configura un experimento;
- muestra resultados;
- genera gráficos;
- compara alternativas;
- documenta el proceso;
- ejecuta una prueba puntual.

## 15. Imports

Preferir:

```python
from slm.ch2_simple_bigram import BigramLanguageModel
```

Evitar:

```python
import sys
sys.path.append("../../src")
```

## 16. Empaquetado

Mantener:

```text
src/
└── slm/
    ├── __init__.py
    ├── ch2_simple_bigram.py
    └── ch2_simple_gpt.py
```

Añadir `pyproject.toml` en la raíz para que:

```bash
pip install -e .
```

haga importable `slm`.

## 17. Nombres de archivos

Para archivos nuevos, preferir:

```text
chapter_02_build_gpt_from_scratch.ipynb
chapter_02_simple_bigram_model.ipynb
```

Evitar, si es posible:

- espacios;
- `:`;
- caracteres especiales.

No renombrar archivos existentes si eso rompe referencias sin necesidad.

## 18. Dependencias

Declarar dependencias preferiblemente en `pyproject.toml`.

No instalar paquetes dentro del código Python salvo en celdas de preparación de Colab.

## 19. GPU

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Comprobar:

```python
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

El código debería poder ejecutar pruebas pequeñas en CPU cuando sea razonable.

## 20. Reproducibilidad

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

Guardar con los resultados los parámetros importantes del experimento.

## 21. Checkpoints

No guardar grandes checkpoints en Git.

Si se generan muchos:

1. escribirlos en `/content`;
2. copiar a Drive sólo los que deban conservarse.

Ejemplo:

```python
from shutil import copy2
from pathlib import Path

local_checkpoint = Path("/content/checkpoint.pt")
persistent_checkpoint = CHECKPOINTS_DIR / "checkpoint.pt"

torch.save(model.state_dict(), local_checkpoint)
copy2(local_checkpoint, persistent_checkpoint)
```

## 22. Antes de usar Colab

1. Terminar el cambio en VS Code.
2. Ejecutar las pruebas locales posibles.
3. Revisar `git status`.
4. Commit.
5. Push.
6. Abrir o ejecutar el notebook.
7. En Colab, hacer `git clone` o `git pull`.
8. Montar Google Drive.
9. Copiar a `/content` los datos usados de forma intensiva.
10. Ejecutar el experimento.
11. Guardar en Drive lo que deba persistir.

## 23. Después de un experimento

Si sólo se han generado resultados:

- guardar resultados o checkpoints en Google Drive;
- no modificar Git.

Si hace falta cambiar código:

1. hacer el cambio preferentemente en VS Code/WSL;
2. probarlo;
3. commit;
4. push;
5. actualizar Colab con `git pull`;
6. volver a ejecutar.

Evitar que Colab se convierta en una copia divergente del código.

## 24. Reglas para Copilot

Al proponer cambios:

1. respetar esta arquitectura;
2. mantener la lógica reutilizable en `src/slm`;
3. mantener notebooks ligeros;
4. no introducir rutas locales de WSL en notebooks destinados a Colab;
5. no versionar ni modificar `notebooks/book/` salvo petición expresa;
6. no copiar código del libro al código propio sin reimplementarlo;
7. mantener compatibilidad CPU cuando sea razonable;
8. diseñar el código para WSL y Colab;
9. evitar duplicar lógica entre `.py` y `.ipynb`;
10. usar GitHub para código;
11. usar Google Drive para datos, modelos, checkpoints, resultados y logs persistentes;
12. usar `/content` para trabajo temporal;
13. evitar lecturas y escrituras intensivas sobre Drive cuando pueda usarse `/content`;
14. preferir cambios pequeños y claros.

## 25. Regla mental

```text
VS Code + WSL
    ↓
desarrollar, probar, refactorizar
    ↓
GitHub
    ↓
sincronizar código
    ↓
Colab
    ↓
GPU / experimentos pesados
    ↓
/content
    ↓
trabajo temporal rápido
    ↓
Google Drive
    ↓
datos, checkpoints, modelos y resultados persistentes
```

La fuente del código es GitHub.

El entorno de desarrollo es VS Code + WSL.

Colab aporta capacidad de cálculo.

`/content` aporta almacenamiento temporal rápido.

Google Drive aporta almacenamiento persistente para datos y artefactos.
