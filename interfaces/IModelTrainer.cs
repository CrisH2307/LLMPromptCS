using Microsoft.ML;

namespace LanguageModelTraining.Interfaces
{
    /// <summary>
    /// Interface for training language models
    /// </summary>
    public interface IModelTrainer
    {
        /// <summary>
        /// Trains a language model on the provided data
        /// </summary>
        /// <param name="trainingData">Processed training data</param>
        /// <param name="epochs">Number of training epochs</param>
        /// <param name="batchSize">Batch size for training</param>
        /// <returns>Trained model transformer</returns>
        ITransformer TrainModel(IDataView trainingData, int epochs = 10, int batchSize = 64);
        
        /// <summary>
        /// Saves a trained model to disk
        /// </summary>
        /// <param name="model">Trained model transformer</param>
        /// <param name="modelPath">Path to save the model</param>
        void SaveModel(ITransformer model, string modelPath);
        
        /// <summary>
        /// Loads a trained model from disk
        /// </summary>
        /// <param name="modelPath">Path to the saved model</param>
        /// <returns>Loaded model transformer</returns>
        ITransformer LoadModel(string modelPath);
    }
}