using LanguageModelTraining.Interfaces;
using LanguageModelTraining.Models;
using Microsoft.ML;
using System;
using System.Linq;

namespace LanguageModelTraining.Services
{
    /// <summary>
    /// Implementation of model evaluation for language models
    /// </summary>
    public class ModelEvaluator : IModelEvaluator
    {
        private readonly MLContext _mlContext;
        
        public ModelEvaluator(MLContext mlContext)
        {
            _mlContext = mlContext;
        }
        
        public ModelMetrics EvaluateModel(ITransformer model, IDataView testData)
        {
            Console.WriteLine("Evaluating model...");
            
            // Transform the test data
            var predictions = model.Transform(testData);
            
            // Evaluate the model
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions);
            
            // Calculate perplexity
            double perplexity = CalculatePerplexity(model, testData);
            
            // Return custom metrics
            return new ModelMetrics
            {
                Loss = metrics.LogLoss,
                Perplexity = perplexity,
                Accuracy = metrics.MicroAccuracy
            };
        }
        
        public double CalculatePerplexity(ITransformer model, IDataView testData)
        {
            // Transform the test data
            var predictions = model.Transform(testData);
            
            // Create an enumerable from the predictions
            var predictionRows = _mlContext.Data.CreateEnumerable<PredictionResult>(predictions, reuseRowObject: false);
            
            // Calculate cross-entropy loss
            double totalLoss = 0;
            int count = 0;
            
            foreach (var prediction in predictionRows)
            {
                if (prediction.Score != null && prediction.Score.Length > 0)
                {
                    // Get the probability of the true label
                    float trueProb = prediction.Score[(int)prediction.Label];
                    
                    // Calculate negative log likelihood
                    double nll = -Math.Log(Math.Max(trueProb, 1e-10));
                    totalLoss += nll;
                    count++;
                }
            }
            
            // Calculate average loss
            double avgLoss = totalLoss / count;
            
            // Perplexity is e^(average negative log likelihood)
            double perplexity = Math.Exp(avgLoss);
            
            return perplexity;
        }
    }
}