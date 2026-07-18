from unittest.mock import patch

try:
    from retrieval_augmented_service.modules import _registry
    from retrieval_augmented_service.modules.database import SOURCE_MODULES as DATABASE
    from retrieval_augmented_service.modules.memory_update import SOURCE_MODULES as MEMORY
    from retrieval_augmented_service.modules.retrieval import SOURCE_MODULES as RETRIEVAL
    from retrieval_augmented_service.modules.scene_encoding import SOURCE_MODULES as ENCODING

    REGISTRY_IMPORT = "retrieval_augmented_service.modules._registry.importlib.import_module"
except ModuleNotFoundError:
    from modules import _registry
    from modules.database import SOURCE_MODULES as DATABASE
    from modules.memory_update import SOURCE_MODULES as MEMORY
    from modules.retrieval import SOURCE_MODULES as RETRIEVAL
    from modules.scene_encoding import SOURCE_MODULES as ENCODING

    REGISTRY_IMPORT = "modules._registry.importlib.import_module"


def test_memory_pipeline_catalog_is_complete():
    assert "connections" in DATABASE
    assert "openvla" in ENCODING
    assert "realtime" in RETRIEVAL
    assert {"backup", "restore", "clear"}.issubset(MEMORY)


def test_registry_defers_real_database_and_model_imports():
    sentinel = object()
    with patch(REGISTRY_IMPORT, return_value=sentinel):
        assert _registry.load({"mock": "mock.backend"}, "mock") is sentinel
