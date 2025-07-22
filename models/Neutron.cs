using System;

namespace LanguageModelTraining
{
    public class Neutron : INeuron
    {
        #region Member Variables
        private NeuralFactor m_factor;
        private double m_biasWeight;
        private double m_error;
        private float m_output;
        #endregion

        #region Properties
        public NeuralFactor Factor 
        { 
            get { return m_factor; }
            set { m_factor = value; }
        }
        
        public double BiasWeight 
        { 
            get { return m_biasWeight; }
            set { m_biasWeight = value; }
        }
        
        public double Error 
        { 
            get { return m_error; }
            set { m_error = value; }
        }
        #endregion 

        #region Methods
        public void Pulse(INeuralLayer layer)
        {
            // Implementation
        }
        
        public void ApplyLearning(INeuralLayer layer)
        {
            // Implementation
        }
        
        public void ReceiveSignal(INeuronSignal signal, INeuronReceptor receptor)
        {
            // Implementation
        }
        
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        #endregion
    }
}