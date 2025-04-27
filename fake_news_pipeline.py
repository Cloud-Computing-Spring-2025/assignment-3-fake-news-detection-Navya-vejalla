from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start Spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# ----- Task 1: Load & Basic Exploration -----

# Load the CSV
data = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create Temporary View
data.createOrReplaceTempView("news_data")

# Show first 5 rows
data.show(5)

# Count total number of articles
print("Total articles:", data.count())

# Retrieve distinct labels
data.select("label").distinct().show()

# Save Task 1 output
data.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# ----- Task 2: Text Preprocessing -----

# Convert text to lowercase
data = data.withColumn("text", lower(col("text")))

# Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words_token")
data_tokenized = tokenizer.transform(data)

# Remove stopwords
remover = StopWordsRemover(inputCol="words_token", outputCol="filtered_words")
data_cleaned = remover.transform(data_tokenized)

# Optional: Create Temporary View
data_cleaned.createOrReplaceTempView("cleaned_news")

# Save Task 2 output
data_cleaned.select("id", "title", "filtered_words", "label").toPandas().to_csv("task2_output.csv", index=False)

# ----- Task 3: Feature Extraction -----

# TF: HashingTF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
featurized_data = hashingTF.transform(data_cleaned)

# IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurized_data)
rescaled_data = idfModel.transform(featurized_data)

# Label Indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
data_indexed = indexer.fit(rescaled_data).transform(rescaled_data)

# Save Task 3 output
data_indexed.select("id", "filtered_words", "features", "label_index").toPandas().to_csv("task3_output.csv", index=False)

# ----- Task 4: Model Training -----

# Split into train and test
train_data, test_data = data_indexed.randomSplit([0.8, 0.2], seed=42)

# Train Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
lr_model = lr.fit(train_data)

# Predictions
predictions = lr_model.transform(test_data)

# Save Task 4 output
predictions.select("id", "title", "label_index", "prediction").toPandas().to_csv("task4_output.csv", index=False)

# ----- Task 5: Evaluate the Model -----

# Evaluators
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy")

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_accuracy.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

# Create Evaluation Results
import pandas as pd

results = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score"],
    "Value": [accuracy, f1]
})

results.to_csv("task5_output.csv", index=False)

print("✅ All tasks completed. Outputs saved for Task 1 to Task 5!")