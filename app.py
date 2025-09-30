from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
import re
from pyhocon import ConfigFactory
import xmltodict
import requests
import numpy as np
import ast
import fnmatch
from langdetect import detect, DetectorFactory
from wordfreq import zipf_frequency

DetectorFactory.seed = 0

app = Flask(__name__)

def connection_to_bitbucket(target_url, token):
    status = False

    # Usar el token que viene del frontend
    if not token:
        print("[!] Error: No se proporcionó token")
        return False, None

    # 2. Usa el token obtenido en la cabecera
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Atlassian-Token": "nocheck"
    }

    try:
        response = requests.get(target_url, headers=headers)
        if response.status_code == 200:
            status = True
        else:
            print(f"[!] Error de conexión: Código de estado {response.status_code}")
            print(f"[!] Respuesta: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Error de red: {e}")
        return False, None

    return status, response


def get_files_of_repository(url_bitbucket, rama, token):
    name_branch = rama
    all_files_list = []
    name_branch_x = "develop" if name_branch.strip() == "" else name_branch.strip()
    
    if check_format_url(url_bitbucket):
        name_project = url_bitbucket.split("/")[-4]
        name_repository = url_bitbucket.split("/")[-2]
        target_api = "https://bitbucket.globaldevtools.bbva.com/bitbucket/rest/api/latest/projects/{}/repos/{}/files?at=refs%2Fheads%2F{}&start=0&limit=1000000".format(
            name_project, name_repository, name_branch_x
        )

        status, response = connection_to_bitbucket(target_api, token)
        if status:
            all_files_list = response.json()['values']
    else:
        print("[!] The URL does not comply with the specified format.")

    return all_files_list

def read_content_file_of_repository(url_bitbucket, filename, name_branch, token):
    content_file = ""
    name_branch_x = "develop" if name_branch.strip() == "" else name_branch.strip()

    name_project = url_bitbucket.split("/")[-4]
    name_repository = url_bitbucket.split("/")[-2]
    target_api = "https://bitbucket.globaldevtools.bbva.com/bitbucket/rest/api/latest/projects/{}/repos/{}/raw/{}?at=refs%2Fheads%2F{}".format(
        name_project, name_repository, filename, name_branch_x
    )
    
    status, response = connection_to_bitbucket(target_api, token)
    if status:
        content_file = response.text
        
    return content_file

def check_format_url(url_bitbucket):
    pattern_url = r"https:\/\/bitbucket\.globaldevtools\.bbva\.com\/bitbucket\/projects\/[\w-]+\/repos\/[\w-]+\/browse"
    if re.match(pattern_url, url_bitbucket):
        return True
    else:
        return False

# Lista de verbos válidos para nombres de funciones
VALID_VERBS = [
    "add", "calculate", "group", "select", "join", "write", "sort",
    "drop", "extract", "check", "append", "is", "concat",
    "filter", "map", "merge", "split", "normalize", "transform",
    "validate", "load", "save", "fetch", "update", "remove",
    "create", "delete", "parse", "build", "clean", "format",
    "union", "run", "read","get", "filter", "cast", "recover", "replace"
]

# Funciones que deben ignorarse en la validación
IGNORED_FUNCS = [
    "setUp", "tearDown", "main", "__init__", "__str__", "__repr__",
    "setUpClass", "tearDownClass"
]

# Patrones de archivos donde se debe validar
VALIDATION_PATTERNS = [
    r"^transformations.*\.py$",   # transformations*.py
    r"^test_transformations.*\.py$",  # test_transformations*.py
    r"^utils\.py$",               # utils.py
    r"^app\.py$"                  # app.py
]

def validate_function_names(file_content, filename):
    bad_funcs = []
    is_test_file = filename.startswith("test_") or "/test" in filename or "\\test" in filename

    try:
        tree = ast.parse(file_content, filename=filename)
    except SyntaxError:
        return bad_funcs

    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # Ignorar la primera función en archivos de test
    if is_test_file and functions:
        functions = functions[1:]

    for node in functions:
        func_name = node.name
        if func_name in IGNORED_FUNCS:
            continue

        parts = func_name.split("_")
        prefix = parts[0]
        func_invalid = False

        if is_test_file:
            # Tests deben empezar con test_ y luego un verbo válido
            if not func_name.startswith("test_"):
                func_invalid = True
            else:
                if len(parts) > 1:
                    verb = parts[1]
                    if verb not in VALID_VERBS:
                        func_invalid = True
                else:
                    func_invalid = True
        else:
            # Verificar si el prefijo es un verbo válido
            if prefix not in VALID_VERBS:
                func_invalid = True

            # Reglas especiales SOLO para estos verbos
            if prefix == "join" and "_with_" not in func_name:
                func_invalid = True
            if prefix == "group" and "_by_" not in func_name:
                func_invalid = True
            if prefix == "filter" and "_by_" not in func_name:
                func_invalid = True
            if prefix == "union" and "_with_" not in func_name:
                func_invalid = True

            # Nota: "run" se permite sin reglas especiales

        if func_invalid:
            bad_funcs.append(func_name)

    return bad_funcs

def check_function_verbs(repository_files, url_repository, rama, token):
    report_blocks = []

    for f in sorted(repository_files, key=lambda x: x.split("/")[-1]):
        fname = f.split("/")[-1]
        if any(re.match(pat, fname) for pat in VALIDATION_PATTERNS):
            content = read_content_file_of_repository(url_repository, f, rama, token)
            bad_funcs = validate_function_names(content, f)

            if bad_funcs:
                bad_funcs = sorted(bad_funcs)
                lines = [f"{i}. {func}" for i, func in enumerate(bad_funcs, 1)]
                block = f"{fname}:\n    " + "\n    ".join(lines)
                report_blocks.append(block)

    return "OK" if not report_blocks else "\n".join(report_blocks)

UDF_IMPORT_PATTERNS = [
    (r"\bfrom\s+pyspark\.sql\.functions\s+import\s+udf\b", "from pyspark.sql.functions import udf"),
    (r"\bpyspark\.sql\.functions\.udf\b", "pyspark.sql.functions.udf"),
    (r"\bimport\s+pyspark\.sql\.functions\s+as\s+\w+\b", "import pyspark.sql.functions as <alias>"),
    (r"\bfrom\s+pyspark\.sql\.types\s+import\s+UserDefinedType\b", "from pyspark.sql.types import UserDefinedType"),
]

def validate_no_udf(file_content, filename):
    hits = []
    lines = file_content.split("\n")

    for line in lines:
        clean = line.strip()
        if clean.startswith("#"):
            continue

        for regex, label in UDF_IMPORT_PATTERNS:
            if re.search(regex, clean):
                hits.append(label)

    if not hits:
        return "OK"

    # armar bloque numerado
    enumerated = [f"{i}. {h}" for i, h in enumerate(hits, 1)]
    return f"{filename}:\n    " + "\n    ".join(enumerated)

def check_no_udf(repository_files, url_repository, rama,token):
    target_patterns = [
        r"^transformations.*\.py$",
        r"^utils\.py$"
    ]

    reports = []
    any_hit = False

    for f in repository_files:
        fname = f.split("/")[-1]
        if any(re.match(pat, fname) for pat in target_patterns):
            content = read_content_file_of_repository(url_repository, f, rama, token)
            result = validate_no_udf(content, fname)
            if result != "OK":
                reports.append(result)
                any_hit = True

    return "OK" if not any_hit else "\n\n".join(reports).strip()

FIELDS_SUFFIX = "_FIELDS"
FIELD_SUFFIX = "_FIELD"

def extract_constants_from_code(code: str):
    constants = {}
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(f"[!] Error parseando fields.py: {e}")
        return constants

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        # Si no se puede evaluar, guardar el tipo
                        if isinstance(node.value, ast.Str):
                            constants[target.id] = node.value.s
                        elif isinstance(node.value, ast.Dict):
                            constants[target.id] = {}
                        elif isinstance(node.value, ast.List):
                            constants[target.id] = []
                        elif isinstance(node.value, ast.Num):
                            constants[target.id] = node.value.n
                        elif isinstance(node.value, ast.NameConstant):
                            constants[target.id] = node.value.value
                        else:
                            constants[target.id] = None
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            try:
                                constants[target.id] = ast.literal_eval(stmt.value)
                            except (ValueError, SyntaxError):
                                constants[target.id] = None
    return constants

def should_be_plural(value):
    if value is None:
        return False

    # Diccionarios siempre son _FIELDS
    if isinstance(value, dict):
        return True

    if isinstance(value, (list, tuple, set)) and len(value) > 1:
        return True

    if isinstance(value, (str, int, float, bool)) or value is None:
        return False

    if isinstance(value, (list, tuple)) and len(value) == 1:
        return False

    return False

def compute_correct_name(name, value):
    should_be_fields = should_be_plural(value)

    # Si ya tiene el sufijo correcto, no cambiar
    if should_be_fields and name.endswith(FIELDS_SUFFIX):
        return name
    elif not should_be_fields and name.endswith(FIELD_SUFFIX):
        return name

    # Remover sufijos incorrectos existentes
    if name.endswith(FIELDS_SUFFIX):
        base_name = name[:-len(FIELDS_SUFFIX)]
    elif name.endswith(FIELD_SUFFIX):
        base_name = name[:-len(FIELD_SUFFIX)]
    else:
        base_name = name

    # Agregar sufijo correcto
    if should_be_fields:
        return f"{base_name}{FIELDS_SUFFIX}"
    else:
        return f"{base_name}{FIELD_SUFFIX}"

def validate_field_naming(constants):
    errors = []

    for name, value in constants.items():
        current_suffix = None
        if name.endswith(FIELDS_SUFFIX):
            current_suffix = "FIELDS"
        elif name.endswith(FIELD_SUFFIX):
            current_suffix = "FIELD"
        else:
            # No tiene sufijo - error
            correct_name = compute_correct_name(name, value)
            errors.append(f"{name} → {correct_name} (falta sufijo)")
            continue

        # Verificar si el sufijo actual es correcto
        should_be_fields = should_be_plural(value)
        correct_suffix = "FIELDS" if should_be_fields else "FIELD"

        if current_suffix != correct_suffix:
            correct_name = compute_correct_name(name, value)
            value_type = type(value).__name__
            if isinstance(value, dict):
                value_info = f"diccionario con {len(value)} elementos"
            elif isinstance(value, (list, tuple, set)):
                value_info = f"colección con {len(value)} elementos"
            else:
                value_info = f"{value_type}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}"

            errors.append(f"{name} → {correct_name} ({value_info})")

    return errors

def validate_fields_field(url_repo, branch,token):
    # Buscar archivos
    files = get_files_of_repository(url_repo, branch,token)
    fields_candidates = [f for f in files if f.endswith("fields.py")]

    if not fields_candidates:
        return "⚠️ No se encontró fields.py en el repo"

    # Leer contenido
    fields_path = fields_candidates[0]
    content = read_content_file_of_repository(url_repo, fields_path, branch, token)
    constants = extract_constants_from_code(content)

    if not constants:
        return "ℹ️ No se encontraron constantes en fields.py"

    errors = validate_field_naming(constants)

    if not errors:
        return "✅ OK"
    else:
        result = f"❌ Se encontraron {len(errors)} errores en la convención FIELD/FIELDS:\n"
        result += "\n".join([f"  • {error}" for error in errors])
        return result

# Función adicional para análisis detallado (opcional)
def analyze_fields_structure(url_repo, branch):
    files = get_files_of_repository(url_repo, branch, token)
    fields_candidates = [f for f in files if f.endswith("fields.py")]

    if not fields_candidates:
        return "⚠️ No se encontró fields.py en el repo"

    fields_path = fields_candidates[0]
    content = read_content_file_of_repository(url_repo, fields_path, branch, token)
    constants = extract_constants_from_code(content)

    analysis = []
    analysis.append(f"📊 Análisis de {fields_path}:")
    analysis.append(f"Total de constantes: {len(constants)}")

    field_count = 0
    fields_count = 0
    no_suffix_count = 0

    for name, value in constants.items():
        if name.endswith(FIELD_SUFFIX):
            field_count += 1
        elif name.endswith(FIELDS_SUFFIX):
            fields_count += 1
        else:
            no_suffix_count += 1

    analysis.append(f"Constantes con _FIELD: {field_count}")
    analysis.append(f"Constantes con _FIELDS: {fields_count}")
    analysis.append(f"Constantes sin sufijo: {no_suffix_count}")

    # Validación
    errors = validate_field_naming(constants)
    if errors:
        analysis.append(f"\n❌ Errores encontrados: {len(errors)}")
        for error in errors[:10]:  # Mostrar solo los primeros 10
            analysis.append(f"  • {error}")
        if len(errors) > 10:
            analysis.append(f"  ... y {len(errors) - 10} más")
    else:
        analysis.append("\n✅ Todas las constantes siguen la convención FIELD/FIELDS")

    return "\n".join(analysis)

def validate_versions(file_kaafile_content, file_setup_cfg_content, file_setup_py_content, file_init_py_content):

    versions = {}

    # setup.cfg
    try:
        #La versión está en la segunda línea
        line = file_setup_cfg_content.splitlines()[1]
        match = re.search(r'=\s*(\S+)', line)
        if match:
            versions['setup.cfg'] = match.group(1).strip()
    except (IndexError, AttributeError):
        versions['setup.cfg'] = "No se encontró la versión"

    #setup.py
    try:
        match = re.search(r'version=["\'](.*?)["\']', file_setup_py_content)
        if match:
            versions['setup.py'] = match.group(1).strip()
    except (AttributeError):
        versions['setup.py'] = "No se encontró la versión"

    # __init__.py
    try:
        match = re.search(r'__version__\s*=\s*["\'](.*?)["\']', file_init_py_content)
        if match:
            versions['__init__.py'] = match.group(1).strip()
    except (AttributeError):
        versions['__init__.py'] = "No se encontró la versión"

    #Kaafile
    try:
        match = re.search(r'version\s*=\s*["\'](.*?)["\']', file_kaafile_content)
        if match:
            versions['Kaafile'] = match.group(1).strip()
    except (AttributeError):
        versions['Kaafile'] = "No se encontró la versión"

    #Comparar las versiones
    version_values = list(versions.values())
    # Eliminar versiones no encontradas
    version_values = [v for v in version_values if "No se encontró" not in v]

    if len(set(version_values)) <= 1 and len(version_values) > 0:
        result = "Vesiones iguales:\n"
        for file, ver in versions.items():
            result += f"- {file}: {ver}\n"
        return result
    else:
        result = "Versiones no consistentes:\n"
        for file, ver in versions.items():
            result += f"- {file}: {ver}\n"
        return result

def validate_no_pandas(content, ruta_archivo):

    nombre_archivo = os.path.basename(ruta_archivo)

    usa_pandas = False
    if re.search(r'^\s*import\s+pandas', content, re.MULTILINE):
        usa_pandas = True
    elif re.search(r'^\s*from\s+pandas', content, re.MULTILINE):
        usa_pandas = True

    return {
        "Archivo": nombre_archivo,
        "Usa pandas": "Sí" if usa_pandas else "No"
    }

def get_variables_short(file_transformations):

    nodo = ast.parse(file_transformations)
    variables = []

    for n in ast.walk(nodo):
        if isinstance(n, ast.Assign):  # solo asignaciones normales
            for target in n.targets:
                if isinstance(target, ast.Name):  # variable simple
                    variables.append(target.id.lower())

                elif isinstance(target, (ast.Tuple, ast.List)):  # asignación múltiple
                    for e in target.elts:
                        if isinstance(e, ast.Name):
                            variables.append(e.id.lower())


    return variables

def busqueda_logica(file_app):
    palabra_1, palabra_2, palabra_3 = "if ", "for ","else "
    if palabra_1 in file_app or palabra_2 in file_app or palabra_3 in file_app:
        resultado = True
    else:
        resultado = False
    return resultado

def busqueda_spark(file):
    palabra = "spark.read."
    if palabra in file:
        resultado = True
    else:
        resultado = False
    return resultado

def busqueda_spark_write(file):
    palabra = ".write."
    if palabra in file:
        # If ".write." is found, now check for "dataproc_mock.write"
        if "dataproc_mock.write" in file:
            return False # Found "dataproc_mock.write", so result is False
        else:
            resultado = True
    else:
        # If ".write." is not found at all
        resultado = False
    return resultado

def busqueda_hadoop(file):
    palabra = "apache.hadoop"
    if palabra in file:
        resultado = True
    else:
        resultado = False
    return resultado

def lecturas_hasta_la_particion(file,file_conf):
    palabra_3 = "="
    file_new = file.replace(" = ","")
    file_new_2 = file_new.replace("==","")
    file_conf_2 = file_conf.replace(" = ","")

    if palabra_3 in file_conf_2:
        resultado = True
    elif palabra_3 in file_new_2:
        resultado = True
    else:
        resultado = False
    return resultado

def version_dataproc_sdk(file_requirements):
    target_version = "0.4.9"
    match = re.search(r"dataproc_sdk==([\d.]+)", file_requirements)
    if match:
        found_version = match.group(1)
        # Convert version strings to comparable tuples of integers
        found_version_parts = tuple(map(int, found_version.split('.')))
        target_version_parts = tuple(map(int, target_version.split('.')))
        if found_version_parts >= target_version_parts:
            resultado = True
        else:
            resultado = False
    else:
        # If dataproc_sdk is not found, consider it incorrect
        resultado = False
    return resultado

def compactacion(file):
    palabra_1 , palabra_2= ".coalesce(", "compact"
    if palabra_1 in file or palabra_2 in file:
        resultado = True
    else:
        resultado = False
    return resultado

def base_path(file):
    comilla_doble=chr(14)
    palabra_1, palabra_2, palabra_3 = ".option(" + comilla_doble + "basepath" + comilla_doble, "base_path","full_path"
    if palabra_1 in file or palabra_2 in file or palabra_3 in file:
        print('usa basepath')
        resultado = True
    else:
        resultado = False
    return resultado

def check_conf_size(file_conf_processed):
    try:
        start_index = file_conf_processed.find('{')
        if start_index != -1:
            relevant_content = file_conf_processed[start_index:]
            conf_char_count = len(relevant_content)
            if conf_char_count > 2400:
                return "Pasó el límite", conf_char_count
            else:
                return "OK", conf_char_count
        else:
            return "application.conf size ERROR: No opening brace found", 0 # Handle cases where no '{' is found
    except Exception as e:
        return f"application.conf size ERROR: {e}", 0

def check_schema_value(file_conf):
    # Parse the HOCON configuration
    try:
        config = ConfigFactory.parse_string(file_conf)
    except Exception as e:
        return f"application.conf SCHEMA ERROR: Could not parse HOCON: {e}"

    # Search for keys containing "SCHEMA" and check their values
    for key, value in config.items():
        if "SCHEMA" in key.upper():
            if isinstance(value, str) and "gl-datio-da-generic-dev" in value:
                return "ERROR: Value contains gl-datio-da-generic-dev"
    return "OK"

def check_sonar_coverage(file_sonar):
    required_line = "sonar.python.coverage.reportPaths=coverage.xml"
    if required_line in file_sonar:
        return "OK"
    else:
        return "FALTA ACTUALIZAR EL SONARQUBE"

def check_input_output_suffixes(file_conf):
    input_found = False
    output_found = False
    # Split the file content into lines and iterate
    for line in file_conf.splitlines():
        # Check if the line contains a variable assignment (contains '=')
        if '=' in line:
            # Extract the variable name (part before the first '=')
            variable_name = line.split('=')[0].strip()
            # Check for suffixes
            if variable_name.upper().endswith('INPUT'):
                input_found = True
            if variable_name.upper().endswith('OUTPUT'):
                output_found = True
    if input_found and output_found:
        return "OK"
    else:
        return "Falta agregar los sufijos Input y Output a las variables"

def check_contributing_readme(repository_files):
    contributing_found = any("CONTRIBUTING" in file for file in repository_files)
    readme_found = any("README" in file for file in repository_files)

    if contributing_found and readme_found:
        return "OK"
    else:
        return "No se encuentran los archivos CONTRIBUTING y/o README"

# Empieza el filtro por idioma
def extract_docstring_params(docstring: str):
    if not docstring:
        return []
    return re.findall(r":param\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:", docstring)

whitelist = {
    "datetime", "utils", "Utils", "concat", "dataframe", "input", "output", "output_df", "filters", "epigraphs",
    "logger", "config", "partition", "result", "input_df", "StringType", "dataproc", "df", "AnalysisException",
    "StructType", "StructField", "relativedelta", "staticmethod", "joined_df", "date_str", "created_df", "DecimalType",
    "DataFrame", "dataframes", "Dataframe", "DoubleType", "persisted_dataframes", "aggregate_func", "unix", "timestamp", "catalog_df",
    "DateType", "irregula", "filtered_nulls_df", "tbuc", "disaggregate", "magicmock", "coment", "parquet", "cond", "guarannteect",
    "comment", "str", "df", "bussiness", "ci", "disaggregated", "input", "df", "fxin", "aggregate", "output", "fxin", "readed",
    "deposit", "emissions", "not", "nmd", "cts", "regexp", "replace", "sum", "amortization", "number", "param", "numerator",
    "pytest", "Constructor", "coalesce", "timedelta", "Dict", "SparkSession", "DataprocExperiment", "DatioPysparkSession",
    "function", "method", "class", "object", "variable", "parameter", "argument", "return", "returns", "value",
    "values", "data", "file", "files", "directory", "path", "paths", "name", "names", "type", "types", "code",
    "string", "strings", "number", "numbers", "integer", "float", "boolean", "array", "list", "lists", "dict",
    "dictionary", "set", "sets", "tuple", "tuples", "key", "keys", "value", "values", "index", "indexes", "item",
    "items", "element", "elements", "length", "size", "count", "total", "sum", "average", "minimum", "maximum",
    "first", "last", "next", "previous", "current", "new", "old", "original", "copy", "clone", "deep", "shallow",
    "null", "none", "true", "false", "yes", "no", "ok", "error", "errors", "exception", "exceptions", "warning",
    "warnings", "info", "information", "debug", "message", "messages", "log", "logs", "logging", "print", "output",
    "input", "source", "target", "destination", "result", "results", "output", "process", "processing", "compute",
    "calculation", "transform", "transformation", "convert", "conversion", "parse", "parsing", "validate", "validation",
    "check", "verify", "test", "testing", "unit", "integration", "system", "acceptance", "quality", "assurance",
    "development", "production", "staging", "testing", "environment", "configuration", "settings", "options",
    "parameters", "arguments", "default", "custom", "special", "general", "specific", "common", "unique", "different",
    "similar", "same", "equal", "equivalent", "identical", "distinct", "separate", "combined", "merged", "joined",
    "split", "divided", "separated", "connected", "linked", "related", "associated", "correlated", "mapped",
    "mapping", "relationship", "connection", "association", "correlation", "dependency", "dependent", "independent",
    "epigraph", "epigraphs", "perimeter", "hierarchies", "taxonomy", "segment", "exchange", "rate", "currency",
    "catalog", "identifier", "identifiers", "merges", "concatenated", "aliased", "substring", "predefined",
    "aliasing", "partitioned", "filtered", "joined", "added", "column", "values", "dict", "hierarchies",
    "offices", "experiment", "session", "mock", "dataproc", "pyspark", "spark", "dataframe", "dataframes"
}

def is_english_word(word: str, min_freq: float = 2.5) -> bool:
    word = word.lower().strip()

    # Verificar whitelist primero
    if word in whitelist:
        return True

    # Ignorar formatos de fecha y códigos
    if re.match(r'^yyyy-MM-dd', word) or re.match(r'^[A-Za-z0-9_]+$', word) and len(word) <= 2:
        return True

    # Ignorar palabras que parecen ser nombres de variables técnicas
    if '_' in word and word.replace('_', '').isalnum():
        return True

    # Verificar frecuencia
    try:
        freq = zipf_frequency(word, 'en')
        return freq >= min_freq
    except:
        # Si hay error en wordfreq, asumir que es inglés
        return True

def is_english_text(text: str) -> bool:
    if not text:
        return True

    # 1. Reemplazar caracteres especiales por espacio
    clean_text = re.sub(r'[_:\.\-,;\*\\/]', ' ', text)

    # 2. Tokenizar
    words = [w.strip() for w in clean_text.split() if w.strip()]

    if not words:
        return True

    # 3. Validar cada palabra con criterios más flexibles
    for word in words:
        word_lower = word.lower()

        # Ignorar palabras muy cortas
        if len(word) <= 2:
            continue

        # Ignorar formatos técnicos
        if (re.match(r'^[A-Z][a-z]+[A-Z]', word) or  # CamelCase
            re.match(r'^[a-z_]+$', word) or          # snake_case
            re.match(r'^[A-Z_]+$', word)):           # CONSTANT_CASE
            continue

        # Verificar si es inglés
        if not is_english_word(word):
            return False

    return True

def validate_functions_in_english(file_content: str, file_path) -> str:
    errors = []
    try:
        tree = ast.parse(file_content)
    except Exception as e:
        return f"ERROR analizando archivo: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            docstring = ast.get_docstring(node)

            if docstring:
                lines = docstring.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if (line and 
                        not line.startswith(':') and 
                        not line.startswith('>>>') and 
                        not line.startswith('...') and  
                        not line.startswith('#') and     
                        len(line) > 10):             
                        
                        if not is_english_text(line):
                            words = line.split()
                            english_words = sum(1 for word in words if is_english_word(word))
                            if english_words < len(words) * 0.6:  # menos del 60% en inglés
                                errors.append(f"documentación línea {i+1}: '{line}'")

            func_params = [arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")]
            for arg in func_params:
                technical_patterns = [
                    arg.endswith('_df'), arg.endswith('_mock'), arg.endswith('_config'),
                    arg.endswith('_dict'), arg.endswith('_list'), arg.endswith('_str'),
                    arg.endswith('_path'), arg.endswith('_file'), arg.endswith('_dir'),
                    '_perimeter_' in arg, '_hierarchies_' in arg, '_epigraph_' in arg,
                    '_exchange_' in arg, '_currency_' in arg, '_catalog_' in arg,
                    '_taxonomy_' in arg, '_rate_' in arg, '_filter_' in arg,
                    arg.startswith('df_'), arg.startswith('mock_'), arg.startswith('config_'),
                    len(arg) <= 3,  
                    arg.replace('_', '').isalnum() and len(arg) > 8 
                ]
                
                if any(technical_patterns):
                    continue  
                    
                if not is_english_text(arg):
                    errors.append(f"parámetro: {arg}")

            local_vars = set()
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Assign):
                    for target in subnode.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            technical_vars = [
                                'df', 'df1', 'df2', 'df3', 'result', 'temp', 'tmp',
                                'i', 'j', 'k', 'v', 'x', 'y', 'z', 'n', 'idx',
                                'col', 'row', 'val', 'item', 'elem', 'obj',
                                'lst', 'dct', 'arr', 'res', 'out', 'input',
                                'config', 'logger', 'spark', 'session', 'mock'
                            ]
                            if (var_name not in technical_vars and 
                                len(var_name) > 4 and  # solo variables significativas
                                not any(var_name.endswith(suffix) for suffix in ['_df', '_mock', '_dict', '_list', '_str']) and
                                not any(pattern in var_name for pattern in ['_perimeter_', '_hierarchies_', '_epigraph_'])):
                                local_vars.add(var_name)

            for var in local_vars:
                if not is_english_text(var):
                    parts = var.split('_')
                    if len(parts) > 1:
                        english_parts = sum(1 for part in parts if is_english_word(part) or len(part) <= 3)
                        if english_parts < len(parts) * 0.7:  
                            errors.append(f"variable: {var}")
                    else:
                        errors.append(f"variable: {var}")

            if ("tests/" not in file_path and 
                docstring and 
                len(docstring.strip()) > 50):  
                
                doc_params = extract_docstring_params(docstring)
                if len(func_params) > 0 and len(doc_params) > 0:
                    missing_in_doc = set(func_params) - set(doc_params)
                    extra_in_doc = set(doc_params) - set(func_params)
                    
                    if len(missing_in_doc) > len(func_params) * 0.5:  # más del 50% faltan
                        errors.append(f"parámetros no documentados: {', '.join(missing_in_doc)}")
                    if len(extra_in_doc) > 3:  # más de 3 parámetros extra
                        errors.append(f"parámetros documentados pero no definidos: {', '.join(extra_in_doc)}")

    if errors:
        enumerated = [f"{idx}. {err}" for idx, err in enumerate(errors, start=1)]
        return "\n".join(enumerated)
    else:
        return "OK"

def validate_multiple_files_in_english(url_repository, rama, file_paths,token):
    results = []
    for file_path in file_paths:
        try:
            file_content = read_content_file_of_repository(url_repository, file_path, rama, token)

            validation_result = validate_functions_in_english(file_content, file_path)
            filename = os.path.basename(file_path)

            # ✅ Solo agregar si hay errores reales (no "OK")
            if not validation_result.startswith("OK"):
                # ELIMINAR el filtro por número de errores
                results.append(f"\n{filename}\n{validation_result}")

        except Exception as e:
            results.append(f"{os.path.basename(file_path)}\nError procesando archivo: {e}")

    return "\n".join(results) if results else "OK"

def check_commented_code(file_content, file_path):
    errors = []
    filename = os.path.basename(file_path)
    is_app_file = "app.py" in file_path
    
    for i, line in enumerate(file_content.splitlines(), start=1):
        stripped = line.strip()
        
        if (stripped.startswith("#") and 
            not stripped.startswith("#!") and 
            "coding" not in stripped.lower()):
            
            if is_app_file:
                if all(c in '# -=' for c in stripped): 
                    continue
                
                line_lower = stripped.lower()
                allowed_phrases = [
                    "your code starts here",
                    "your code ends here", 
                    "inicio del código",
                    "fin del código",
                    "start here",
                    "end here"
                ]
                if any(phrase in line_lower for phrase in allowed_phrases):
                    continue
            
            errors.append(f"Línea {i}: {stripped[:80]}")

    if errors:
        enumerated = [f"{idx}. {err}" for idx, err in enumerate(errors, start=1)]
        return f"Comentarios invalidos {filename}:\n" + "\n".join(enumerated)
    else:
        return f"{filename} - ok"

def validate_commented_code_in_files(url_repository, rama, file_paths,token):
    results = []
    for file_path in file_paths:
        try:
            file_content = read_content_file_of_repository(url_repository, file_path, rama, token)

            comment_check = check_commented_code(file_content, file_path)
            results.append(comment_check)
        except:
            results.append(f"{file_path}\nArchivo no encontrado")

    return "\n".join(results)

def data_frame(url_repository, rama, token):
    repository_files = get_files_of_repository(url_repository, rama, token)
    name_project = url_repository.split("/")[-4]
    name_repository = url_repository.split("/")[-2]

    # files de interés
    files_analysis = ["/utils/", "/transformations/", "/experiment.py", "/app.py"]
    files = []
    for file in repository_files:
        [files.append(file) for i in files_analysis if i in file]

    file_aplicattion = 'resources/application.conf'
    file_requirements = 'requirements.txt'
    file_app = name_repository+'/app.py'
    file_sonar_properties = 'sonar-project.properties'
    file_constants = name_repository+'/constants.py'
    file_utils = name_repository+'/utils/utils.py'
    file_fields = name_repository+'/fields.py'
    file_constants = name_repository+'/constants.py'
    file_test_utils = 'tests/utils/test_utils.py'
    file_test_app = 'tests/test_app.py'

    # ARCHIVOS PARA CHEQUEAR VERSIONES
    file_kaafile = 'Kaafile'
    file_setup_cfg = 'setup.cfg'
    file_setup_py = 'setup.py'
    file_init_py =  name_repository+'/__init__.py'

    # Variables de estado
    x1 = True
    x2 = True
    x3 = True
    x4 = True
    x5 = True
    x6 = True

    file_conf = read_content_file_of_repository(url_repository, file_aplicattion, rama, token)
    file_conf_processed = file_conf.replace('\n', '\r\n')

    # Procesar archivos para análisis de Spark, Hadoop, etc.
    for fil in files:
        file = read_content_file_of_repository(url_repository, fil, rama, token)

       # Spark
        resultado_spark = busqueda_spark(file)
        if resultado_spark and x1:
            result_spark = "❌ Lectura con Spark"
            x1 = False
        elif not resultado_spark and x1:
            result_spark = "✅ No se está usando lectura con Spark"

        # Hadoop
        resultado_hadoop = busqueda_hadoop(file)
        if resultado_hadoop and x2:
            result_hadoop = "❌ Uso de hadoop"
            x2 = False
        elif not resultado_hadoop and x2:
            result_hadoop = "✅ No se está usando hadoop"

        # Lectura hasta la particion
        resultado_lect_part = lecturas_hasta_la_particion(file, file_conf)
        if resultado_lect_part and x3:
            result_lec_part = "✅ Lectura hasta el Objeto"
            x3 = False
        elif not resultado_lect_part and x3:
            result_lec_part = "❌ Lectura hasta la particion"

        # Escritura con Spark
        resultado_spark_write = busqueda_spark_write(file)
        if resultado_spark_write and x4:
            result_write = "❌ Escritura con Spark"
            x4 = False
        elif not resultado_spark_write and x4:
            result_write = "✅ No se está escribiendo con Spark"

        # Uso de Compactacion
        resultado_comp = compactacion(file)
        if resultado_comp and x5:
            result_compac = "✅ Se esta usando compactacion"
            x5 = False
        elif not resultado_comp and x5:
            result_compac = "❌ No se esta usando compactacion"

        # Uso de base_path
        resultado_base_path = base_path(file)
        if resultado_base_path and x6:
            result_basepath = "❌ Se esta usando base_path"
            x6 = False
        elif not resultado_base_path and x6:
            result_basepath = "✅ No se esta usando base_path"


   # Version correcta de dataproc_sdk
    file_requirements = read_content_file_of_repository(url_repository, file_requirements, rama, token)
    resultado_dataproc = version_dataproc_sdk(file_requirements)
    if resultado_dataproc:
        result_dataproc = "✅ Version correcta de dataproc"
    else:
        result_dataproc = "❌ Version incorrecta de dataproc"

    # Uso de lógica en app.py
    file_app = read_content_file_of_repository(url_repository, file_app, rama, token)
    resultado_busqlogicaapp = busqueda_logica(file_app)
    if resultado_busqlogicaapp:
        result_busqlogicaapp = "❌ Sí se está haciendo uso de lógica (if, for, else)"
    else:
        result_busqlogicaapp = "✅ No se detectó lógica compleja"

    files_transformations = [
        file for file in repository_files
        if f"{name_repository}/transformations/" in file and "transformations" in file.split("/")[-1]
    ]

    files_to_validate_main = files_transformations + [
        f"{name_repository}/utils/utils.py",
        f"{name_repository}/app.py",
        f"{name_repository}/fields.py",
        f"{name_repository}/constants.py",
    ]

    files_to_validate_tests = [
        "tests/transformations/test_transformations.py",
        "tests/utils/test_utils.py",
        "tests/test_app.py",
    ]

    files_to_validate = files_to_validate_main + files_to_validate_tests

    english_validation_general = validate_multiple_files_in_english(url_repository, rama, files_to_validate, token)
    resultado_pr = validate_commented_code_in_files(url_repository, rama, files_to_validate, token)
    verbos_output = check_function_verbs(repository_files, url_repository, rama, token)
    conf_check_field_fields = validate_fields_field(url_repository, rama, token)
    udf_output = check_no_udf(repository_files, url_repository, rama, token)

    file_transformations = [
        f for f in repository_files
        if fnmatch.fnmatch(f, "*/transformations/transformations*.py")
    ]

    file_test_transformations = [
        x for x in repository_files
        if fnmatch.fnmatch(x, "*tests/transformations/test_transformations*.py")
    ]

    file_paths_verb_ex = [
        file_utils,
        file_app,
        file_fields,
        file_constants,
        file_test_utils,
        file_test_app
    ]

    file_paths_verb = [*file_transformations, *file_test_transformations, *file_paths_verb_ex]

    ListVerb = [
        "add", "calculate", "group", "select", "join", "write", "sort",
        "drop", "extract", "check", "append", "is", "concat",
        "filter", "map", "merge", "split", "normalize", "transform",
        "validate", "load", "save", "fetch", "update", "remove",
        "create", "delete", "parse", "build", "clean", "format",
        "union", "run", "read","get", "filter", "cast", "recover", "replace"
    ]

    matches_by_file = {}
    for ruta_verb in file_paths_verb:
        content = read_content_file_of_repository(url_repository, ruta_verb, rama, token)
        file_name = os.path.basename(ruta_verb)
        variables = get_variables_short(content)
        variables_match = [var for var in variables if var.split("_")[0] in ListVerb]
        if variables_match:
            matches_by_file[file_name] = variables_match

    salida = ""
    for archivo, variables in matches_by_file.items():
        salida += f"\n{archivo}:\n"
        for idx, var in enumerate(variables, start=1):
            salida += f"  {idx}. {var}\n"

    # Validación de Pandas
    file_paths_pandas = [*file_transformations, *file_test_transformations, file_utils]
    result_pandas_re = {}
    for ruta_pandas in file_paths_pandas:
        content = read_content_file_of_repository(url_repository, ruta_pandas, rama, token)
        result_pandas = validate_no_pandas(content, ruta_pandas)
        result_pandas_re[result_pandas["Archivo"]] = result_pandas["Usa pandas"]

    result_pandas_str = ""
    for archivo, estado in result_pandas_re.items():
        icon = "❌" if estado == "Sí" else "✅"
        result_pandas_str += f"{icon} {archivo}: {estado}\n"

    # Otras validaciones
    conf_size_check, conf_char_count = check_conf_size(file_conf_processed)
    schema_value_check = check_schema_value(file_conf)

    file_sonar = read_content_file_of_repository(url_repository, file_sonar_properties, rama, token)
    sonar_coverage_check = check_sonar_coverage(file_sonar)

    input_output_check = check_input_output_suffixes(file_conf)
    contributing_readme_check = check_contributing_readme(repository_files)

    file_constants_content = read_content_file_of_repository(url_repository, file_constants, rama, token)


    # Check versions
    file_kaafile_content = read_content_file_of_repository(url_repository, file_kaafile, rama, token)
    file_setup_cfg_content = read_content_file_of_repository(url_repository, file_setup_cfg, rama, token)
    file_setup_py_content = read_content_file_of_repository(url_repository, file_setup_py, rama, token)
    file_init_py_content = read_content_file_of_repository(url_repository, file_init_py, rama, token)
    resultado_validacion_versiones = validate_versions(file_kaafile_content, file_setup_cfg_content, file_setup_py_content, file_init_py_content)

     # ========== PRESENTACIÓN MEJORADA ==========
    print("=" * 80)
    print(f"📊 ANÁLISIS DEL REPOSITORIO: {name_repository}")
    print("=" * 80)

    # Configuración y Entorno
    print("\n🔧 CONFIGURACIÓN Y ENTORNO")
    print("-" * 40)
    print(f"📖 Lectura de datos: {result_spark}")
    print(f"✍️  Escritura de datos: {result_write}")
    print(f"🐘 Uso de Hadoop: {result_hadoop}")
    print(f"📊 Nivel de lectura: {result_lec_part}")
    print(f"🔗 Versión Dataproc SDK: {result_dataproc}")
    print(f"🗜️  Compactación: {result_compac}")
    print(f"📁 Base Path: {result_basepath}")
    print(f"🔍 Lógica en app.py: {result_busqlogicaapp}")

    # Archivos de Configuración
    print("\n📁 ARCHIVOS DE CONFIGURACIÓN")
    print("-" * 40)
    icon_size = "✅" if "OK" in conf_size_check else "❌"
    print(f"📏 Tamaño application.conf: {icon_size} {conf_size_check} ({conf_char_count} caracteres)")

    icon_schema = "✅" if "OK" in schema_value_check else "❌"
    print(f"🏷️  Schema productivo: {icon_schema} {schema_value_check}")

    icon_sonar = "✅" if "OK" in sonar_coverage_check else "❌"
    print(f"📊 Configuración Sonar: {icon_sonar} {sonar_coverage_check}")

    icon_io = "✅" if "OK" in input_output_check else "❌"
    print(f"🔤 Sufijos Input/Output: {icon_io} {input_output_check}")

    icon_docs = "✅" if "OK" in contributing_readme_check else "❌"
    print(f"📚 Documentación: {icon_docs} {contributing_readme_check}")

    # Calidad de Código - MEJORADO
    print("\n🎯 CALIDAD DE CÓDIGO")
    print("=" * 40)

    # Organizar en subsecciones
    print("\n📝 CONVENCIONES DE CÓDIGO")
    print("-" * 30)

    # Verbos en funciones
    if verbos_output == "OK":
        print("🔤 Verbos en funciones: ✅ Correctos")
    else:
        print("🔤 Verbos en funciones: ❌ Problemas detectados")
        if verbos_output != "OK":
            print("\n   📋 FUNCIONES IDENTIFICADAS:")
            print("   " + "=" * 50)
            
            # Procesar y formatear la salida
            lines = verbos_output.split('\n')
            current_file = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detectar si es un archivo nuevo
                if line.endswith(':') or ':' in line and not any(char.isdigit() for char in line.split(':')[0]):
                    if ':' in line:
                        file_part = line.split(':')[0]
                        current_file = file_part.strip()
                        print(f"\n   📁 {current_file}:")
                    else:
                        current_file = line.replace(':', '').strip()
                        print(f"\n   📁 {current_file}:")
                        
                # Detectar líneas con funciones numeradas
                elif any(char.isdigit() for char in line) and ('.' in line or line.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9'))):
                    # Separar las funciones numeradas
                    functions = []
                    current_func = ""
                    
                    # Procesar la línea para separar funciones
                    parts = line.split()
                    i = 0
                    while i < len(parts):
                        if parts[i].endswith('.') and parts[i][:-1].isdigit():
                            # Es un número, empezar nueva función
                            if current_func:
                                functions.append(current_func.strip())
                            current_func = parts[i] + " " + (parts[i+1] if i+1 < len(parts) else "")
                            i += 2
                        else:
                            current_func += " " + parts[i]
                            i += 1
                    
                    if current_func:
                        functions.append(current_func.strip())
                    
                    # Imprimir funciones formateadas
                    for func in functions:
                        if func.strip():
                            print(f"     {func}")
                            
                # Para líneas que son claramente nombres de archivo sin ":"
                elif any(ext in line for ext in ['.py', '.py:']):
                    current_file = line.replace(':', '').strip()
                    print(f"\n   📁 {current_file}:")
                    
                else:
                    # Línea normal de texto
                    print(f"     {line}")

    # Fields
    if conf_check_field_fields == "✅ OK":
        print("🏷️  Fields/FIELDS: ✅ Correctos")
    else:
        print("🏷️  Fields/FIELDS: ❌ Problemas detectados")
        if conf_check_field_fields != "✅ OK":
            print("\n   ⚠️  ERRORES ENCONTRADOS:")
            print("   " + "-" * 30)
            lines = conf_check_field_fields.split('\n')
            for line in lines:
                if line.strip():
                    if "–" in line:  # Línea con formato FIELD – FIELD
                        field_error = line.strip()
                        print(f"\n   🔸 {field_error}")
                    elif "Se encontraron" in line or "errores" in line:
                        print(f"\n   📊 {line.strip()}")

    print("\n🌎 IDIOMA Y DOCUMENTACIÓN")
    print("-" * 30)

    # Idiomas
    if english_validation_general == "OK":
        print("🗣️  Idioma: ✅ Todo en inglés")
    else:
        print("🗣️  Idioma: ❌ Problemas detectados")
        if english_validation_general != "OK":
            for line in english_validation_general.split('\n'):
                if line.strip():
                    if not line.startswith(' ') and not line.startswith('1.'):
                        print(f"   📄 {line}")
                    else:
                        print(f"     {line}")

    print("\n🚫 RESTRICCIONES TÉCNICAS")
    print("-" * 30)

    # Uso de Pandas
    pandas_errors = sum(1 for line in result_pandas_str.split('\n') if "❌" in line)
    if pandas_errors == 0:
        print("🐼 Uso de Pandas: ✅ Correcto")
    else:
        print(f"🐼 Uso de Pandas: ❌ {pandas_errors} archivos usan Pandas")
        # Mostrar solo los archivos problemáticos
        for line in result_pandas_str.split('\n'):
            if "❌" in line and line.strip():
                print(f"   {line}")

    # UDF
    if udf_output == "OK":
        print("⚡ UDF: ✅ No se detectaron UDFs")
    else:
        print("⚡ UDF: ❌ Se detectaron UDFs")
        lines = udf_output.split('\n')
        for i, line in enumerate(lines[:3]):  # Mostrar solo primeros 3
            if line.strip():
                print(f"   {i+1}. {line}")

    comment_errors = len([line for line in resultado_pr.split('\n') if "Comentarios invalidos" in line])
    if comment_errors == 0:
        print("💬 Código comentado: ✅ Correcto")
    else:
        print(f"💬 Código comentado: ❌ {comment_errors} archivos con problemas")
        for line in resultado_pr.split('\n'):
            if line.strip():
                if "Comentarios invalidos" in line:
                    print(f"   📄 {line}")
                elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    print(f"     {line}")

    print("\n🔢 CONTROL DE VERSIONES")
    print("-" * 30)

    version_lines = resultado_validacion_versiones.split('\n')
    if version_lines and "iguales" in version_lines[0].lower():
        print("📦 Versiones: ✅ Consistentes")
        for line in version_lines[1:]:  # Mostrar detalles
            if line.strip() and line.startswith('-'):
                print(f"   {line}")
    else:
        print("📦 Versiones: ❌ Inconsistentes")
        for line in version_lines:
            if line.strip():
                print(f"   {line}")

    if salida.strip():
        print("\n⚠️  VARIABLES CON PATRÓN DE VERBOS")
        print("-" * 40)
        files_with_verbs = len([line for line in salida.split('\n') if line.endswith(':')])
        print(f"Se encontraron {files_with_verbs} archivos con variables que siguen patrón de verbos")
        # MOSTRAR TODOS los archivos y variables
        for line in salida.split('\n'):
            if line.strip():
                if line.endswith(':'):
                    print(f"   📁 {line}")
                elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    print(f"     {line}")

    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)

    d = {
        "engine": [name_repository],
        "Lectura": [result_spark],
        "Uso_Hadoop": [result_hadoop],
        "Lectura_Particion": [result_lec_part],
        "version_dataproc": [result_dataproc],
        "escritura": [result_write],
        "Uso de compactacion": [result_compac],
        "Uso base_path": [result_basepath],
        "Se hace uso de lógica en el app.py?": [result_busqlogicaapp],
        "application.conf Caracteres": [conf_size_check],
        "Cant. Caracteres": [conf_char_count],
        "SCHEMA Productivo": [schema_value_check],
        "Archivo Sonar": [sonar_coverage_check],
        "Input/Output Var. Entorno": [input_output_check],
        "CONTRIBUTING/README": [contributing_readme_check],
        "Palabra con idioma inválido": [english_validation_general],
        "Nombre de función correcta": [verbos_output],
        "Field y Fields Incorrectos": [conf_check_field_fields],
        "Nombre de campo/variable incorrectos": [salida],
        "Uso de Pandas": [result_pandas_str],
        "Validación UDF": [udf_output],
        "Codigo Comentado Inválido (###)": [resultado_pr],
        "Resultado de validación de versiones": [resultado_validacion_versiones]
    }

    return pd.DataFrame(data=d)


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 Analizador de Repositorios Bitbucket</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: #2c3e50;
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .form-section {
                padding: 30px;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #2c3e50;
                font-size: 14px;
            }
            
            input, textarea {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            input:focus, textarea:focus {
                border-color: #4285f4;
                outline: none;
                box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.1);
            }
            
            .button-group {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                margin-top: 30px;
            }
            
            button {
                background: #4285f4;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            button:hover {
                background: #3367d6;
                transform: translateY(-2px);
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            
            .btn-secondary {
                background: #6c757d;
            }
            
            .btn-secondary:hover {
                background: #545b62;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
                background: #f8f9fa;
                border-radius: 10px;
                margin: 20px 0;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #4285f4;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .resultado {
                display: none;
                margin-top: 30px;
                padding: 0;
                border-radius: 10px;
                border: 1px solid #e9ecef;
            }
            
            .result-header {
                background: #28a745;
                color: white;
                padding: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .result-content {
                padding: 25px;
                max-height: 600px;
                overflow-y: auto;
            }
            
            .section {
                margin-bottom: 25px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #4285f4;
            }
            
            .section h3 {
                color: #2c3e50;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .metric-card {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 6px;
                border: 1px solid #e9ecef;
            }
            
            .estado-ok { color: #28a745; font-weight: bold; }
            .estado-error { color: #dc3545; font-weight: bold; }
            .estado-warning { color: #ffc107; font-weight: bold; }
            
            .help-text {
                font-size: 12px;
                color: #6c757d;
                margin-top: 5px;
                font-style: italic;
            }
            
            .required {
                color: #dc3545;
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                }
                
                .header h1 {
                    font-size: 2em;
                }
                
                .button-group {
                    flex-direction: column;
                }
                
                button {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Analizador de Repositorios Bitbucket</h1>
                <p>Analiza automáticamente la calidad de código de tus repositorios</p>
            </div>
            
            <div class="form-section">
                <div class="form-group">
                    <label for="url">URL del Repositorio Bitbucket <span class="required">*</span></label>
                    <input type="url" id="url" placeholder="https://bitbucket.globaldevtools.bbva.com/bitbucket/projects/..." required>
                    <div class="help-text">URL completa del repositorio de Bitbucket</div>
                </div>
                
                <div class="form-group">
                    <label for="rama">Rama a analizar <span class="required">*</span></label>
                    <input type="text" id="rama" placeholder="develop" value="develop" required>
                    <div class="help-text">Nombre de la rama que deseas analizar</div>
                </div>
                
                <div class="form-group">
                    <label for="token">Token de Bitbucket</label>
                    <input type="password" id="token" placeholder="Ingresa tu token si es necesario">
                </div>

                <div class="button-group">
                    <button onclick="analizar()" id="btnAnalizar">
                        <span>🔍</span> Ejecutar Análisis
                    </button>
                    <button onclick="limpiar()" class="btn-secondary">
                        <span>🔄</span> Limpiar
                    </button>
                </div>
            </div>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                <h3>⏳ Analizando repositorio...</h3>
                <p>Esto puede tomar unos momentos mientras revisamos todos los archivos</p>
            </div>

            <div id="resultado" class="resultado">
                <!-- Los resultados se cargan aquí dinámicamente -->
            </div>
        </div>

        <script>
            function analizar() {
                const url = document.getElementById('url').value.trim();
                const rama = document.getElementById('rama').value.trim();
                const token = document.getElementById('token').value.trim();
                
                if (!url) {
                    alert('❌ Por favor ingresa una URL válida');
                    return;
                }
                
                if (!rama) {
                    alert('❌ Por favor ingresa el nombre de la rama');
                    return;
                }
                
                // Mostrar loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultado').style.display = 'none';
                document.getElementById('btnAnalizar').disabled = true;
                
                // Enviar petición
                fetch('/analizar', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        url: url,
                        rama: rama,
                        token: token
                    })
                })
                .then(response => response.json())
                .then(data => {
                    mostrarResultado(data);
                })
                .catch(error => {
                    mostrarError(error);
                })
                .finally(() => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('btnAnalizar').disabled = false;
                });
            }
            
            function mostrarResultado(data) {
                const resultadoDiv = document.getElementById('resultado');
                
                if (data.success) {
                    let html = `
                        <div class="result-header">
                            <h2><span>✅</span> Análisis Completado</h2>
                            <span>${new Date().toLocaleString()}</span>
                        </div>
                        <div class="result-content">    
                    `;
                    
                    // Mostrar datos del análisis
                    if (data.data && data.columns) {
                        html += '<div class="section"><h3>📊 Resultados del Análisis</h3>';
                        
                        data.columns.forEach((columna, index) => {
                            if (data.data[columna] && data.data[columna][0]) {
                                const valor = data.data[columna][0];
                                const estado = getEstadoFromValor(valor);
                                
                                html += `
                                    <div class="metric-card">
                                        <strong>${columna}:</strong>
                                        <span class="${estado.clase}">${valor}</span>
                                    </div>
                                `;
                            }
                        });
                        
                        html += '</div>';
                    }
                    
                    html += `</div></div>`;
                    resultadoDiv.innerHTML = html;
                } else {
                    mostrarError(data.error);
                }
                
                resultadoDiv.style.display = 'block';
            }
            
            function mostrarError(error) {
                const resultadoDiv = document.getElementById('resultado');
                const errorMsg = typeof error === 'string' ? error : error.message || 'Error desconocido';
                
                resultadoDiv.innerHTML = `
                    <div class="result-header" style="background: #dc3545;">
                        <h2><span>❌</span> Error en el Análisis</h2>
                    </div>
                    <div class="result-content">
                        <div class="section">
                            <h3>⚠️ Detalles del Error</h3>
                            <div class="metric-card">
                                <strong>Error:</strong>
                                <span class="estado-error">${errorMsg}</span>
                            </div>
                            <p>Verifica que la URL sea correcta y que tengas acceso al repositorio.</p>
                        </div>
                    </div>
                `;
                resultadoDiv.style.display = 'block';
            }
            
            function limpiar() {
                document.getElementById('url').value = '';
                document.getElementById('rama').value = 'develop';
                document.getElementById('token').value = '';
                document.getElementById('resultado').style.display = 'none';
                document.getElementById('resultado').innerHTML = '';
            }
            
            function getEstadoFromValor(valor) {
                const texto = String(valor || ""); // fuerza a string incluso si es null/undefined
                if (texto.includes('✅') || texto.includes('OK')) {
                    return { clase: 'estado-ok' };
                } else if (texto.includes('❌') || texto.includes('ERROR')) {
                    return { clase: 'estado-error' };
                } else if (texto.includes('⚠️') || texto.includes('WARNING')) {
                    return { clase: 'estado-warning' };
                }
                return { clase: '' };
            }
            
            document.addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    analizar();
                }
            });
        </script>
    </body>
    </html>
    '''

# === API ENDPOINTS ===
@app.route('/analizar', methods=['POST'])
def analizar():
    try:
        data = request.get_json()
        url = data.get("url")
        rama = data.get("rama", "develop")
        token = data.get("token")
        
        if not url or not token:
            return jsonify({"success": False, "error": "URL y Token son requeridos"}), 400
        
        print(f"🔍 Analizando: {url} (rama: {rama})")
        
        # Usar tu función corregida
        resultado = data_frame(url, rama, token)
        
        # Convertir para el frontend
        if hasattr(resultado, 'to_dict'):
            df_dict = resultado.to_dict('list')
            response_data = {
                "success": True,
                "data": df_dict,
                "columns": list(resultado.columns),
                "url": url,
                "rama": rama
            }
        else:
            response_data = {
                "success": True,
                "data": {"resultado": str(resultado)},
                "url": url, 
                "rama": rama
            }
            
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)