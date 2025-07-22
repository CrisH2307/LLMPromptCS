using Microsoft.ML;
using LanguageModelApi.Models;
using System;
using System.IO;
using System.Threading.Tasks;

namespace LanguageModelApi.Services
{
    public class LanguageModelService
    {
        private readonly MLContext _mlContext;
        private readonly IConfiguration _configuration;
        private readonly AdvancedTextGenerator _textGenerator;
        private ITransformer? _model;
        private string _modelPath;
        private bool _isModelLoaded = false;
        
        public LanguageModelService(MLContext mlContext, IConfiguration configuration)
        {
            _mlContext = mlContext;
            _configuration = configuration;
            _modelPath = _configuration["LanguageModel:ModelPath"] ?? "models/my-model/model.zip";
            _textGenerator = new AdvancedTextGenerator(mlContext);
        }
        
        public async Task<string> GenerateTextAsync(string prompt, int maxLength = 100, float temperature = 0.7f, float topP = 0.9f)
        {
            // Load model if not already loaded
            if (!_isModelLoaded)
            {
                await LoadModelAsync();
            }
            
            if (_model == null)
            {
                throw new InvalidOperationException("Model could not be loaded");
            }
            
            // Create prediction engine
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(_model);
            
            // Make prediction
            var prediction = predictionEngine.Predict(new TextData { Text = prompt });
            
            // Generate text based on the prediction
            string generatedText = GenerateTextFromPrediction(prompt, prediction, maxLength, temperature, topP);
            
            return generatedText;
        }
        
        private async Task LoadModelAsync()
        {
            if (!File.Exists(_modelPath))
            {
                throw new FileNotFoundException($"Model file not found at {_modelPath}");
            }
            
            try
            {
                _model = _mlContext.Model.Load(_modelPath, out var _);
                _isModelLoaded = true;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to load model: {ex.Message}", ex);
            }
            
            await Task.CompletedTask; // Just to make the method async
        }
        
        private string GenerateTextFromPrediction(string prompt, TransformedTextData prediction, int maxLength, float temperature, float topP)
        {
            if (prediction.Tokens != null && prediction.Tokens.Length > 0)
            {
                // Use the advanced text generator
                return _textGenerator.GenerateWithNucleusSampling(
                    prompt,
                    prediction.Tokens,
                    temperature,
                    topP,
                    maxLength);
            }
            
            // Fallback if no tokens are available
            return $"{prompt} → [No tokens generated]";
        }
    }
}