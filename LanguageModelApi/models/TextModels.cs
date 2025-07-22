using System.Text.Json.Serialization;
using Microsoft.ML.Data;

namespace LanguageModelApi.Models
{
    public class TextData
    {
        public string? Text { get; set; }
    }

    public class TransformedTextData
    {
        [JsonIgnore]
        public float[]? Features { get; set; }
        
        public string[]? Tokens { get; set; }
    }

    public class GenerateTextRequest
    {
        public string Prompt { get; set; } = "";
        public int MaxLength { get; set; } = 100;
        public float Temperature { get; set; } = 0.7f;
        public float TopP { get; set; } = 0.9f;
        public bool IncludeTokens { get; set; } = false;
    }

    public class GenerateTextResponse
    {
        public string Prompt { get; set; } = "";
        public string GeneratedText { get; set; } = "";
        public string[]? Tokens { get; set; }
        public GenerationMetadata? Metadata { get; set; }
    }

    public class GenerationMetadata
    {
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public float Temperature { get; set; }
        public float TopP { get; set; }
        public int MaxLength { get; set; }
        public int GeneratedLength { get; set; }
        public double ProcessingTimeMs { get; set; }
    }
}