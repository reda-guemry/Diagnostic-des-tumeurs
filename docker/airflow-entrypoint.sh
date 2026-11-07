#!/bin/sh
set -eu

export AIRFLOW_HOME="${AIRFLOW_HOME:-/app/.airflow}"
mkdir -p "$AIRFLOW_HOME"

# Airflow standalone reads DAGs from $AIRFLOW_HOME/dags. Keep the project DAG
# folder as the single source of truth instead of copying DAG files around.
if [ ! -d "$AIRFLOW_HOME/dags" ]; then
    rm -f "$AIRFLOW_HOME/dags"
    ln -s /app/airflow/dags "$AIRFLOW_HOME/dags"
fi

exec "$@"
