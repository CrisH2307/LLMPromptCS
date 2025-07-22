namespace LanguageModelTraining
{
    public interface INeuronReceptor
    {
        float Weight { get; set; }
        void ReceiveSignal(INeuronSignal signal, NeuralFactor factor);
    }
}