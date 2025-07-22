using LanguageModelTraining.Interfaces;
using LanguageModelTraining.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace LanguageModelTraining.Services
{
    /// <summary>
    /// Implementation of data processing for language model training
    /// </summary>
    public class DataProcessor : IDataProcessor
    {
        private readonly MLContext _mlContext;
        private Dictionary<string, int> _vocabulary;
        private Dictionary<int, string> _reverseVocabulary;
        
        public DataProcessor(MLContext mlContext)
        {
            _mlContext = mlContext;
            _vocabulary = new Dictionary<string, int>();
            _reverseVocabulary = new Dictionary<int, string>();
        }
        
        public IDataView LoadAndPrepareData(string dataPath)
        {
            if (!File.Exists(dataPath))
            {
                throw new FileNotFoundException($"Data file not found: {dataPath}");
            }
            
            // Load raw text data
            var data = _mlContext.Data.LoadFromTextFile<TextData>(
                dataPath,
                separatorChar: '\t',
                hasHeader: false);
            
            // If vocabulary is empty, build it
            if (_vocabulary.Count == 0)
            {
                BuildVocabulary(dataPath);
            }
            
            // Define the data processing pipeline
            var pipeline = _mlContext.Transforms.CustomMapping<TextData, TokenizedText>(
                (input, output) => {
                    // Tokenize the input text
                    var tokens = TokenizeText(input.Text);
                    output.TokenIds = tokens.Take(tokens.Length - 1).ToArray();
                    output.NextTokenId = new float[] { tokens.Last() };
                },
                contractName: "TextToTokensMapping");
            
            // Apply the pipeline to the data
            var transformedData = pipeline.Fit(data).Transform(data);
            
            return transformedData;
        }
        
        public float[] TokenizeText(string text)
        {
            if (string.IsNullOrEmpty(text))
            {
                return Array.Empty<float>();
            }
            
            // Simple whitespace tokenization for demonstration
            // In a real implementation, you would use a more sophisticated tokenizer
            var tokens = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
            
            // Convert tokens to IDs
            var tokenIds = new List<float>();
            foreach (var token in tokens)
            {
                var normalizedToken = token.ToLowerInvariant();
                if (_vocabulary.TryGetValue(normalizedToken, out int id))
                {
                    tokenIds.Add(id);
                }
                else
                {
                    // Use unknown token ID (0)
                    tokenIds.Add(0);
                }
            }
            
            return tokenIds.ToArray();
        }
        
        public int BuildVocabulary(string dataPath)
        {
            Console.WriteLine($"Building vocabulary from {dataPath}...");
            
            // Reset vocabulary
            _vocabulary.Clear();
            _reverseVocabulary.Clear();
            
            // Add special tokens
            _vocabulary["<unk>"] = 0;  // Unknown token
            _vocabulary["<pad>"] = 1;  // Padding token
            _vocabulary["<sos>"] = 2;  // Start of sequence
            _vocabulary["<eos>"] = 3;  // End of sequence
            
            // Reverse mapping
            _reverseVocabulary[0] = "<unk>";
            _reverseVocabulary[1] = "<pad>";
            _reverseVocabulary[2] = "<sos>";
            _reverseVocabulary[3] = "<eos>";
            
            int nextId = 4;
            
            // Read all text and build vocabulary
            foreach (var line in File.ReadLines(dataPath))
            {
                var tokens = line.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var token in tokens)
                {
                    var normalizedToken = token.ToLowerInvariant();
                    if (!_vocabulary.ContainsKey(normalizedToken))
                    {
                        _vocabulary[normalizedToken] = nextId;
                        _reverseVocabulary[nextId] = normalizedToken;
                        nextId++;
                    }
                }
            }
            
            Console.WriteLine($"Vocabulary built with {_vocabulary.Count} tokens");
            return _vocabulary.Count;
        }
        
        public string DecodeTokens(float[] tokenIds)
        {
            var sb = new StringBuilder();
            foreach (var id in tokenIds)
            {
                int tokenId = (int)id;
                if (_reverseVocabulary.TryGetValue(tokenId, out string token))
                {
                    sb.Append(token).Append(' ');
                }
                else
                {
                    sb.Append("<unk> ");
                }
            }
            return sb.ToString().Trim();
        }
    }
}