import langchain_experimental
print(f"Version: {langchain_experimental.__version__}")
try:
    from langchain_experimental.text_splitters import SemanticChunker
    print("Import successful from text_splitters")
except ImportError as e:
    print(f"Failed from text_splitters: {e}")

try:
    from langchain_experimental.text_splitter import SemanticChunker
    print("Import successful from text_splitter")
except ImportError as e:
    print(f"Failed from text_splitter: {e}")

import pkgutil
print("Submodules:", [name for _, name, _ in pkgutil.iter_modules(langchain_experimental.__path__)])
