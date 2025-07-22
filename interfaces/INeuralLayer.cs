namespace LanguageModelTraining
{
    public interface INeuralLayer
    {
        INeuron[] Neurons { get; }
        INeuralNet Network { get; }
        void ConnectTo(INeuralNet network);
    }
}