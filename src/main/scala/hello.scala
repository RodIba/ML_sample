import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml 
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.io.Source
import org.apache.spark.ml.linalg.Vectors 
import org.apache.spark.mllib.linalg.{Vectors => SparkVectors}
import org.apache.spark.ml.Pipeline
import java.sql.Date
import java.io.File
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel,DecisionTreeClassifier}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

case class Cuartos(id:Int, date: Date,  temp: Double, Humidity: Double, CO2: Double, 
                   Light: Double,HimidityRatio: Double, Occupancy: Int)

object Hello{
  
  def main(args: Array[String]) {
  
    val spark = SparkSession.builder().master("local[*]").appName("ML_Test").getOrCreate()
    import spark.implicits._
    
    spark.sparkContext.setLogLevel("ERROR")
    
		// Import Data
		//val resource = getClass.getClassLoader.getResource("datatest.txt")
		//val path = new File(resource.toURI).getPath
		//println(path)
		
		val traindata = spark.read.format("csv").option("header","true")
		                    .schema(Encoders.product[Cuartos].schema)
		                    .load("/home/rodrigo/Spark/prueba/src/main/resources/datatraining.txt")
		                    .cache()
		                    
		val testdata = spark.read.format("csv").option("header", "true")
		                  .schema(Encoders.product[Cuartos].schema)
		                  .load("/home/rodrigo/Spark/prueba/src/main/resources/datatest.txt")
		                  .cache()
		
		val train_data = traindata.select($"temp", $"Humidity", $"Light", $"CO2", $"Occupancy")
		                          
		val test_data = testdata.select($"temp", $"Humidity", $"Light", $"CO2", $"Occupancy")
		
		// Logistic Regression
		val assembler = (new VectorAssembler()
                   .setInputCols(Array("temp", "Humidity", "Light", "CO2"))
                   .setOutputCol("predOccupancy"))
                   
		val logRegMod = new LogisticRegression()
                        .setMaxIter(10)
                        .setFeaturesCol("predOccupancy")
                        .setLabelCol("Occupancy")
                        .setFamily("binomial")
		
		val pipe = new Pipeline().setStages(Array(assembler, logRegMod))
		
		val model = pipe.fit(train_data)
		val results = model.transform(test_data)
		
		//results.show()
		// Linear Regression
		val lin_train_data = traindata.select($"Light", $"CO2", $"Humidity")
		                          
		val lin_test_data = testdata.select($"Light", $"CO2", $"Humidity")
		
    
		val lin_assembler = new VectorAssembler()
                            .setInputCols(Array("Light", "Humidity"))
                            .setOutputCol("predCO2")
                            
    val LinRegMod = new LinearRegression().setMaxIter(10)
                                          .setFeaturesCol("predCO2")
                                          .setLabelCol("CO2")
 
    val lin_pipe = new Pipeline().setStages(Array(lin_assembler, LinRegMod))
    val lin_model = lin_pipe.fit(lin_train_data)
    val lin_restuls = lin_model.transform(lin_test_data)
    
    //lin_restuls.show()
    
    //Decision Trees 
    /*
    val datos = spark.sparkContext.textFile("/home/rodrigo/Spark/prueba/src/main/resources/datatraining.txt")
    val header = datos.first()
    val dato = datos.filter(x => x != header)
    
    val datosCsv = dato.map(x => x.split(","))
    
    val dataPoints = datosCsv.map(row => 
          new LabeledPoint(
          row.last.toDouble, 
          SparkVectors.dense(row.take(row.length - 1).map(str => str.toDouble))
    )
  ).cache()
  
  
    val tree_model = DecisionTree.train(dataPoints) 
     */

      val dtree = new DecisionTreeClassifier()                   
                    .setFeaturesCol("predOccupancy")
                    .setLabelCol("Occupancy")
                    
      val dtree_pipe = new Pipeline().setStages(Array(assembler, dtree))
      val dtree_model = dtree_pipe.fit(train_data)
      val dtree_res = dtree_model.transform(test_data)
      
      // Gradient Boosted Decision Trees 
      
      val GB_dtree = new GBTClassifier()
                        .setLabelCol("Occupancy")
                        .setFeaturesCol("predOccupancy")
                        
      val GB_pipe = new Pipeline().setStages(Array(assembler, GB_dtree))
      val GB_model = GB_pipe.fit(train_data)
      val GB_results = GB_model.transform(test_data)
      
      GB_results.show()
      val evaluator = new MulticlassClassificationEvaluator()
                          .setLabelCol("Occupancy")
                          .setPredictionCol("prediction")
                          .setMetricName("accuracy")
                          
      val accuracy_tree = evaluator.evaluate(dtree_res)    
      val accuracy_reg = evaluator.evaluate(results)
      val accuracy_GB = evaluator.evaluate(GB_results)
      
      println(accuracy_tree, accuracy_reg, accuracy_GB)         
      
      
		spark.stop()
  }
}
