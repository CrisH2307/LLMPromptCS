using Microsoft.ML.Data;

namespace LanguageModelTraining.Models
{
    /// <summary>
    /// Class representing raw text data for input
    /// </summary>
    public class TextData
    {
        [LoadColumn(0)]
        public string Text { get; set; } = string.Empty;
    }

    /// <summary>
    /// Class representing tokenized text data
    /// </summary>
    public class TokenizedText
    {
        [VectorType]
        public float[] TokenIds { get; set; } = Array.Empty<float>();
        
        [VectorType]
        public float[] NextTokenId { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// Class representing model prediction output
    /// </summary>
    public class TokenPrediction
    {
        [VectorType]
        public float[] NextTokenPredictions { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// Class representing model evaluation metrics
    /// </summary>
    public class ModelMetrics
    {
        public double Loss { get; set; }
        public double Perplexity { get; set; }
        public double Accuracy { get; set; }
    }
}