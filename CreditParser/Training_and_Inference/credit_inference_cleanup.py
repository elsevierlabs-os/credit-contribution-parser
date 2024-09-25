# Databricks notebook source
import pyspark.sql.functions as f

# COMMAND ----------

credit_df = table("database.table")

# COMMAND ----------

filter_empty = credit_df.filter(f.col('inference_output') != f.array())

# COMMAND ----------

extract_auid = filter_empty.withColumn('auid', f.element_at(f.col('output_exploded'), 1))

# COMMAND ----------

groupby_auid_df = extract_auid.groupby('PII', 'Au', 'auid').agg(f.struct('auid', f.flatten(f.collect_set('inference_output')).alias('contributions')).alias('credit_contributions'))

# COMMAND ----------

final_df = (groupby_auid_df.groupby('PII', 'Au').agg(f.collect_set('credit_contributions').alias('credit_contributions')))

# COMMAND ----------

final_df.write.mode("overwrite").format("delta").saveAsTable("database.table")