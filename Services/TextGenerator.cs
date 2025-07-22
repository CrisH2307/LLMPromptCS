using LanguageModelTraining.Interfaces;
using LanguageModelTraining.Models;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace LanguageModelTraining.Services
{
    /// <summary>
    /// Implementation of text generation using a trained language model
    /// </summary>
    public class TextGenerator : ITextGenerator
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly IDataProcessor _dataProcessor;
        private readonly Random _random;
        
        public TextGenerator(MLContext mlContext, ITransformer model)
        {
            _mlContext = mlContext;
            _model = model;
            _dataProcessor = new DataProcessor(mlContext);
            _random = new Random();
        }
        
        public string GenerateText(string seedText, int maxLength = 100, float temperature = 0.7f)
        {
            if (string.IsNullOrEmpty(seedText))
            {
                throw new ArgumentException("Seed text cannot be empty", nameof(seedText));
            }
            
            Console.WriteLine($"Generating text with seed: '{seedText}'");
            
            // Tokenize the seed text
            var tokens = _dataProcessor.TokenizeText(seedText);
            var generatedTokens = new List<float>(tokens);
            
            // Create prediction engine
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<TokenizedText, TokenPrediction>(_model);
            
            // Generate text token by token
            for (int i = 0; i < maxLength; i++)
            {
                // Get the context window (last N tokens)
                var contextWindow = generatedTokens.Skip(Math.Max(0, generatedTokens.Count - 50)).ToArray();
                
                // Predict next token
                var prediction = predictionEngine.Predict(new TokenizedText { TokenIds = contextWindow });
                
                // Apply temperature to adjust randomness
                var probabilities = ApplyTemperature(prediction.NextTokenPredictions, temperature);
                
                // Sample from the distribution
                float nextToken = SampleFromDistribution(probabilities);
                
                // Add the token to generated sequence
                generatedTokens.Add(nextToken);
                
                // Check if we generated an end of sequence token
                if (nextToken == 3) // <eos> token
                {
                    break;
                }
            }
            
            // Convert tokens back to text
            return (_dataProcessor as DataProcessor).DecodeTokens(generatedTokens.ToArray());
        }
        
        public string CompleteText(string partialText, int maxNewTokens = 20)
        {
            // This is similar to GenerateText but with a specific limit on new tokens
            var fullText = GenerateText(partialText, maxNewTokens);
            
            // Return only the completed part (remove the seed text)
            return fullText.Substring(partialText.Length).Trim();
        }
        
        private float[] ApplyTemperature(float[] logits, float temperature)
        {
            if (temperature <= 0)
            {
                throw new ArgumentException("Temperature must be positive", nameof(temperature));
            }
            
            // Apply softmax with temperature
            var scaled = logits.Select(x => x / temperature).ToArray();
            var maxVal = scaled.Max();
            var expVals = scaled.Select(x => Math.Exp(x - maxVal)).ToArray();
            var sum = expVals.Sum();
            
            return expVals.Select(x => (float)(x / sum)).ToArray();
        }
        
        private float SampleFromDistribution(float[] probabilities)
        {
            // Sample a token based on the probability distribution
            var cumulative = 0.0;
            var sample = _random.NextDouble();
            
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (sample < cumulative)
                {
                    return i;
                }
            }
            
            // Fallback to the most likely token
            return Array.IndexOf(probabilities, probabilities.Max());
        }
    }
}