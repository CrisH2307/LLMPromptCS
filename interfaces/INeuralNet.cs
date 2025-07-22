namespace LanguageModelTraining
{
    public interface INeuralNet
    {
        INeuralLayer[] Layers { get; }
        void Process(float[] input);
        float[] GetOutput();
    }
}