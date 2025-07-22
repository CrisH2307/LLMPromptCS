using Microsoft.ML;
using LanguageModelTraining.Models;

namespace LanguageModelTraining.Interfaces
{
    /// <summary>
    /// Interface for evaluating language models
    /// </summary>
    public interface IModelEvaluator
    {
        /// <summary>
        /// Evaluates a trained model on test data
        /// </summary>
        /// <param name="model">Trained model transformer</param>
        /// <param name="testData">Test data for evaluation</param>
        /// <returns>Evaluation metrics</returns>
        ModelMetrics EvaluateModel(ITransformer model, IDataView testData);
        
        /// <summary>
        /// Calculates perplexity of the model on test data
        /// </summary>
        /// <param name="model">Trained model transformer</param>
        /// <param name="testData">Test data for evaluation</param>
        /// <returns>Perplexity score (lower is better)</returns>
        double CalculatePerplexity(ITransformer model, IDataView testData);
    }
}