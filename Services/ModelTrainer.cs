using LanguageModelTraining.Interfaces;
using LanguageModelTraining.Models;
using Microsoft.ML;
using System;
using System.IO;

namespace LanguageModelTraining.Services
{
    /// <summary>
    /// Implementation of model training for language models
    /// </summary>
    public class ModelTrainer : IModelTrainer
    {
        private readonly MLContext _mlContext;
        
        public ModelTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }
        
        public ITransformer TrainModel(IDataView trainingData, int epochs = 10, int batchSize = 64)
        {
            Console.WriteLine($"Training model with {epochs} epochs and batch size {batchSize}...");
            
            // Define the training pipeline
            // In a real implementation, this would be a more sophisticated neural network
            // For demonstration, we'll use a simple logistic regression model
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "Label",
                    inputColumnName: nameof(TokenizedText.NextTokenId))
                .Append(_mlContext.Transforms.Concatenate(
                    outputColumnName: "Features",
                    nameof(TokenizedText.TokenIds)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    maximumNumberOfIterations: epochs))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));
            
            // Train the model
            Console.WriteLine("Fitting the model...");
            var model = pipeline.Fit(trainingData);
            Console.WriteLine("Model training completed");
            
            return model;
        }
        
        public void SaveModel(ITransformer model, string modelPath)
        {
            // Ensure directory exists
            string? directory = Path.GetDirectoryName(modelPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            
            // Save the model
            _mlContext.Model.Save(model, null, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");
        }
        
        public ITransformer LoadModel(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Model file not found: {modelPath}");
            }
            
            // Load the model
            var model = _mlContext.Model.Load(modelPath, out var _);
            Console.WriteLine($"Model loaded from {modelPath}");
            
            return model;
        }
    }
}