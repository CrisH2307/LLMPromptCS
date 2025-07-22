using Microsoft.ML;

namespace LanguageModelTraining.Interfaces
{
    /// <summary>
    /// Interface for language model implementations
    /// </summary>
    public interface ILanguageModel
    {
        /// <summary>
        /// Initializes the language model architecture
        /// </summary>
        /// <param name="vocabularySize">Size of the vocabulary</param>
        /// <param name="embeddingSize">Size of token embeddings</param>
        /// <param name="hiddenSize">Size of hidden layers</param>
        void Initialize(int vocabularySize, int embeddingSize, int hiddenSize);
        
        /// <summary>
        /// Predicts the next token given a sequence of input tokens
        /// </summary>
        /// <param name="inputTokens">Sequence of input token IDs</param>
        /// <returns>Probability distribution over the vocabulary for the next token</returns>
        float[] PredictNextToken(float[] inputTokens);
        
        /// <summary>
        /// Gets the model's transformer that can be used for predictions
        /// </summary>
        ITransformer GetTransformer();
    }
}