using Microsoft.ML;
using Microsoft.ML.Data;

namespace LanguageModelTraining.Interfaces
{
    /// <summary>
    /// Interface for processing text data for language model training
    /// </summary>
    public interface IDataProcessor
    {
        /// <summary>
        /// Loads and prepares data from a file for training or evaluation
        /// </summary>
        /// <param name="dataPath">Path to the data file</param>
        /// <returns>Processed data ready for training or evaluation</returns>
        IDataView LoadAndPrepareData(string dataPath);
        
        /// <summary>
        /// Tokenizes text into a sequence of tokens
        /// </summary>
        /// <param name="text">Input text to tokenize</param>
        /// <returns>Array of token IDs</returns>
        float[] TokenizeText(string text);
        
        /// <summary>
        /// Builds vocabulary from a corpus of text
        /// </summary>
        /// <param name="dataPath">Path to the corpus file</param>
        /// <returns>Number of tokens in vocabulary</returns>
        int BuildVocabulary(string dataPath);
    }
}