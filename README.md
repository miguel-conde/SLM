# SLM

Experimentos con modelos de lenguaje pequeños. El código reutilizable vive en
[src/slm/](src/slm/) y los experimentos propios en
[notebooks/mios/](notebooks/mios/). WSL/local es el entorno de desarrollo;
Colab aporta cálculo remoto temporal y Google Drive, almacenamiento de datos,
checkpoints y resultados. No sincronizar el repositorio mediante Drive.

## Dependencias y alcance de la validación

[pyproject.toml](pyproject.toml) declara el paquete instalable desde `src/`,
la versión de PyTorch y el extra `dev` con pytest. Es la única fuente de
versiones directas: no copiar la versión de PyTorch en comandos o notebooks.
[pytest.ini](pytest.ini) configura la búsqueda de pruebas y `src/`.

Se conserva `requires-python = ">=3.12"`. No se ha encontrado una justificación
documentada ni una matriz de pruebas que permita decidir si ese mínimo fue
intencional. La prueba se ha validado con Python 3.12.10 en Linux/WSL.
Los metadatos de las versiones instaladas de PyTorch y pytest permiten
Python >=3.10, pero eso **no demuestra** compatibilidad del proyecto con
Python 3.10/3.11 ni con todas las versiones posteriores a 3.12.

Las dependencias directas están fijadas, pero no hay un lockfile de dependencias
transitivas ni del backend de construcción. El flujo reproduce la instalación
de las versiones directas y la prueba, no un entorno idéntico bit a bit.
Requiere conexión para instalar; la prueba no descarga datos ni modelos.

## WSL/local: entorno aislado y PyTorch para CPU

Desde la raíz del repositorio, con Python 3.12 y soporte para `venv` disponibles:

```bash
python3.12 -m venv .venv

TORCH_REQUIREMENT="$(.venv/bin/python -I -c \
  'import tomllib; print(next(d for d in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"] if d.startswith("torch==")))')"

.venv/bin/python -m pip --isolated install \
  --index-url https://download.pytorch.org/whl/cpu "$TORCH_REQUIREMENT"
.venv/bin/python -m pip --isolated install -e '.[dev]'

.venv/bin/python -I -m pytest -q
.venv/bin/python -m pip --isolated check
```

No usar `sudo`, `pip` global ni `--system-site-packages`. No hace falta activar
el entorno: se invoca su Python explícitamente. `pip --isolated` evita que la
configuración personal de pip cambie los índices utilizados.
Seleccionar `.venv/bin/python` como intérprete en VS Code.

El orden importa: primero se instala la rueda CPU desde el índice oficial de
PyTorch; después, el paquete editable y las dependencias de desarrollo desde
el índice predeterminado de pip. La rueda CPU satisface la versión declarada
en el proyecto. No usar `--no-deps`, pues ocultaría dependencias necesarias.
Estos pasos están pensados para un entorno nuevo o ya preparado para CPU:
si contiene una rueda CUDA de la misma versión, pip puede conservarla.
La comprobación siguiente detecta ese caso; no se debe dar el entorno por
validado si falla.

### Comprobar procedencia y aislamiento

Desde la raíz:

```bash
.venv/bin/python -I - <<'PY'
from pathlib import Path
import site
import sys

import pytest
import slm
import torch

root = Path.cwd().resolve()
venv = root / ".venv"
assert Path(sys.prefix).resolve() == venv
assert sys.prefix != sys.base_prefix
assert not site.ENABLE_USER_SITE
assert "include-system-site-packages = false" in (
    venv / "pyvenv.cfg"
).read_text()
for module in (pytest, torch):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(venv), path
    print(module.__name__, module.__version__, path)
assert Path(slm.__file__).resolve() == root / "src/slm/__init__.py"
assert torch.version.cuda is None, "Se esperaba una rueda CPU"
print("Python:", sys.executable)
print("slm:", slm.__file__)
PY

git check-ignore .venv/pyvenv.cfg
git status --short
```

El entorno, las cachés y los metadatos de instalación quedan fuera de Git.
La instalación mínima puede emitir una advertencia de PyTorch por ausencia
de NumPy: la prueba no utiliza conversiones a NumPy y no lo necesita.
La advertencia no equivale a un fallo, pero no debe confundirse con una
validación de esas conversiones.

## Google Colab: runtime con GPU

Este flujo es distinto del entorno local. Seleccionar un runtime con GPU en
Colab. Su sistema de archivos no ve el de WSL y `/content` es temporal.
Las instalaciones siguientes son exclusivamente para el runtime desechable
de Colab, no para el Python global de WSL/local.

El repositorio remoto debe contener los archivos de configuración y pruebas.
Los cambios locales sin publicar no llegan con `git clone` o `git pull`.
Para reproducir un experimento, elegir y registrar una revisión publicada.
No se publica ni se crea ningún commit automáticamente con estos pasos.

En un runtime nuevo:

```python
!git clone https://github.com/miguel-conde/SLM.git /content/SLM
%cd /content/SLM
```

Si ya está clonado, no repetir el clon:

```python
%cd /content/SLM
!git pull --ff-only
```

Con el árbol de trabajo limpio, seleccionar la revisión publicada deseada
(sustituir el marcador) y comprobar Python:

```python
!git checkout --detach <COMMIT_PUBLICADO>
!git rev-parse HEAD
import sys
print(sys.version)
```

El runtime debe satisfacer `requires-python`. Si no lo cumple, detenerse:
no rebajar el requisito sin validar compatibilidad.

Instalar en el intérprete del notebook, **sin el índice de CPU local**:

```python
%pip install -e ".[dev]"
%pip check
```

En Linux compatible, la distribución predeterminada de PyTorch puede incluir
soporte CUDA; esto no garantiza compatibilidad con cualquier GPU o driver.
Si se reemplazó PyTorch ya importado, reiniciar la sesión de Python antes de
continuar y volver a ejecutar `%cd /content/SLM`.

Comprobar la GPU explícitamente en la nueva sesión:

```python
import torch

print("PyTorch:", torch.__version__)
print("Importado desde:", torch.__file__)
print("CUDA de la rueda:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
assert torch.version.cuda is not None, "La rueda no tiene soporte CUDA"
assert torch.cuda.is_available(), "El runtime no puede usar CUDA"
print("GPU:", torch.cuda.get_device_name(0))
```

Si falla, no presentar la instalación como apta para GPU. Consultar el
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/) para
elegir un índice CUDA compatible con el runtime y la versión declarada.
No sustituirlo por el índice CPU. Para reinstalar desde el índice elegido
sin duplicar la versión del proyecto:

```python
import subprocess
import sys
import tomllib

with open("pyproject.toml", "rb") as file:
    dependencies = tomllib.load(file)["project"]["dependencies"]
torch_requirement = next(d for d in dependencies if d.startswith("torch=="))
cuda_index = "https://download.pytorch.org/whl/<INDICE_CUDA_COMPATIBLE>"
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--force-reinstall",
    "--index-url", cuda_index, torch_requirement,
])
```

Sustituir el marcador solo tras comprobar la disponibilidad de esa versión.
Después, reiniciar Python y repetir las comprobaciones de CUDA y `pip check`.
Si no existe una rueda compatible, detenerse y registrar el bloqueo; no
cambiar silenciosamente la versión del proyecto. Un runtime de Colab trae
otros paquetes preinstalados, que pueden ser incompatibles con el pin del
proyecto: cualquier conflicto de `pip check` queda pendiente de resolver.

Ejecutar la prueba desde la raíz, utilizando el intérprete del notebook:

```python
import subprocess
import sys

subprocess.check_call([sys.executable, "-I", "-m", "pytest", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "check"])
```

La prueba sigue usando CPU y datos sintéticos incluso en Colab. Que pase
**no valida** entrenamiento en GPU: para eso se requieren las comprobaciones
CUDA y el experimento correspondiente. Este flujo de Colab está documentado,
pero no se ha ejecutado ni certificado en un runtime remoto.

Guardar datos y resultados persistentes en Google Drive, no el repositorio
ni el entorno virtual. No es necesario montar Drive para ejecutar esta prueba.
