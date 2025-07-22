using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace LanguageModelApi.Services
{
    public class AdvancedTextGenerator
    {
        private readonly MLContext _mlContext;
        private readonly Random _random;
        
        public AdvancedTextGenerator(MLContext mlContext)
        {
            _mlContext = mlContext;
            _random = new Random();
        }
        
        public string GenerateWithNucleusSampling(string prompt, string[] tokens, float temperature = 0.7f, float topP = 0.9f, int maxLength = 100)
        {
            // Start with the prompt
            var result = new List<string> { prompt };
            
            // If no tokens provided, return just the prompt
            if (tokens == null || tokens.Length == 0)
            {
                return prompt;
            }
            
            // Get unique tokens for our vocabulary
            var vocabulary = tokens.Distinct().ToArray();
            
            // Simple n-gram based generation (this is a simplified approach)
            for (int i = 0; i < maxLength; i++)
            {
                // Get the last word of the current result
                string lastWord = result.Last().Split(' ').Last();
                
                // Find possible next words (tokens that start with similar letters)
                var possibleNextWords = vocabulary
                    .Where(t => !string.IsNullOrEmpty(t) && t != lastWord)
                    .Select(t => new { Token = t, Score = CalculateSimilarityScore(lastWord, t) })
                    .Where(x => x.Score > 0)
                    .OrderByDescending(x => x.Score)
                    .ToList();
                
                if (!possibleNextWords.Any())
                {
                    // No matching words found, pick a random one
                    result.Add(vocabulary[_random.Next(vocabulary.Length)]);
                    continue;
                }
                
                // Apply temperature to adjust randomness
                var adjustedScores = ApplyTemperature(possibleNextWords.Select(x => x.Score).ToArray(), temperature);
                
                // Apply nucleus (top-p) sampling
                string nextWord = ApplyNucleusSampling(
                    possibleNextWords.Select(x => x.Token).ToArray(),
                    adjustedScores,
                    topP);
                
                result.Add(nextWord);
                
                // Check if we've generated a sentence ending
                if (nextWord.EndsWith('.') || nextWord.EndsWith('!') || nextWord.EndsWith('?'))
                {
                    // 30% chance to end generation after a sentence
                    if (_random.NextDouble() < 0.3)
                    {
                        break;
                    }
                }
            }
            
            return string.Join(" ", result);
        }
        
        private float CalculateSimilarityScore(string word1, string word2)
        {
            // This is a very simple similarity heuristic
            // In a real implementation, you would use the model's predictions
            
            // Check for prefix match
            if (word2.StartsWith(word1.Substring(0, Math.Min(2, word1.Length)), StringComparison.OrdinalIgnoreCase))
            {
                return 0.5f;
            }
            
            // Check for common letters
            int commonChars = word1.ToLower().Intersect(word2.ToLower()).Count();
            return (float)commonChars / Math.Max(word1.Length, word2.Length);
        }
        
        private float[] ApplyTemperature(float[] scores, float temperature)
        {
            if (temperature <= 0)
            {
                // Just return the highest score with 1.0
                var result = new float[scores.Length];
                int maxIndex = Array.IndexOf(scores, scores.Max());
                result[maxIndex] = 1.0f;
                return result;
            }
            
            // Apply temperature scaling
            var scaledScores = scores.Select(s => MathF.Pow(s, 1.0f / temperature)).ToArray();
            
            // Normalize to get probabilities
            float sum = scaledScores.Sum();
            return scaledScores.Select(s => s / sum).ToArray();
        }
        
        private string ApplyNucleusSampling(string[] tokens, float[] probabilities, float topP)
        {
            // Sort tokens by probability in descending order
            var sortedIndices = Enumerable.Range(0, probabilities.Length)
                .OrderByDescending(i => probabilities[i])
                .ToArray();
            
            // Calculate cumulative probabilities
            float cumulativeProb = 0;
            var nucleusIndices = new List<int>();
            
            foreach (var idx in sortedIndices)
            {
                cumulativeProb += probabilities[idx];
                nucleusIndices.Add(idx);
                
                if (cumulativeProb >= topP)
                {
                    break;
                }
            }
            
            // Sample from the nucleus
            float sample = (float)_random.NextDouble();
            float cumulative = 0;
            
            // Renormalize probabilities within the nucleus
            float nucleusSum = nucleusIndices.Sum(i => probabilities[i]);
            
            foreach (var idx in nucleusIndices)
            {
                float normalizedProb = probabilities[idx] / nucleusSum;
                cumulative += normalizedProb;
                
                if (sample <= cumulative)
                {
                    return tokens[idx];
                }
            }
            
            // Fallback to the highest probability token
            return tokens[sortedIndices[0]];
        }
    }
}