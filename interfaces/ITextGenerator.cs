namespace LanguageModelTraining.Interfaces
{
    /// <summary>
    /// Interface for generating text using a trained language model
    /// </summary>
    public interface ITextGenerator
    {
        /// <summary>
        /// Generates text starting from a seed text
        /// </summary>
        /// <param name="seedText">Initial text to start generation from</param>
        /// <param name="maxLength">Maximum length of generated text</param>
        /// <param name="temperature">Sampling temperature (higher = more random)</param>
        /// <returns>Generated text</returns>
        string GenerateText(string seedText, int maxLength = 100, float temperature = 0.7f);
        
        /// <summary>
        /// Completes a partial text using the model
        /// </summary>
        /// <param name="partialText">Text to complete</param>
        /// <param name="maxNewTokens">Maximum number of new tokens to generate</param>
        /// <returns>Completed text</returns>
        string CompleteText(string partialText, int maxNewTokens = 20);
    }
}