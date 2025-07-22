namespace LanguageModelTraining
{
    public interface INeuron
    {
        void ReceiveSignal(INeuronSignal signal, INeuronReceptor receptor);
        void Pulse(INeuralLayer layer);
        void ApplyLearning(INeuralLayer layer);
        NeuralFactor Factor { get; set; }
    }
}