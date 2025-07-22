using Microsoft.ML.Data;

namespace LanguageModelTraining.Models
{
    public class PredictionResult
    {
        public uint Label { get; set; }
        
        [VectorType]
        public float[] Score { get; set; }
        
        public uint PredictedLabel { get; set; }
    }
}