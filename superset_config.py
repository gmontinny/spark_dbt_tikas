from superset.config import *  # noqa

FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
}

# Aumenta timeout para queries longas no Trino/Spark
SQLLAB_TIMEOUT = 300
SUPERSET_WEBSERVER_TIMEOUT = 300
