# Run a local Spark workflow

Install the optional Spark dependency and ensure a compatible Java runtime is
available:

```bash
python -m pip install "M_Ana_package[big]"
```

```python
from MAna.big import deduplicate_latest, snake_case_columns, spark_session

with spark_session("orders-cleanup", master="local[2]", log_level="WARN") as spark:
    orders = spark.read.parquet("orders/")
    orders = snake_case_columns(orders)
    latest = deduplicate_latest(orders, ["order_id"], "updated_at")
    latest.write.mode("overwrite").parquet("clean-orders/")
```

Leave `master` unset in cluster deployments so the launcher remains in control.
Use `repartition_dataframe` before expensive keyed work or partitioned writes;
use `coalesce` only when reducing partitions without partition columns.
