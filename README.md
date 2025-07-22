
# Language Model Training in C#

This project demonstrates how to train and use language models (both small and large) using C# and ML.NET. It provides a structured approach to text processing, model training, evaluation, and text generation.

<img width="1352" height="932" alt="Screenshot 2025-07-22 at 12 26 38 PM" src="https://github.com/user-attachments/assets/f0ede780-f920-4558-a559-64c2beb4d514" />

## Project Structure

- **Program.cs**: Main entry point for the application
- **Interfaces/**: Contains interface definitions for all components
- **Models/**: Contains data structures and model implementations
- **Services/**: Contains implementations of data processing, training, evaluation, and text generation
- **Data/**: Contains training and test data

## Features

- Text data processing and tokenization
- Language model training using ML.NET
- Model evaluation with metrics like perplexity and loss
- Text generation using trained models
- Support for both small and large language models

## Getting Started

### Prerequisites

- .NET 7.0 SDK or later
- ML.NET packages (included in the project file)

### Building the Project

```bash
dotnet build
```

### Running the Project

```bash
dotnet run
```

## How It Works

1. **Data Processing**: The `DataProcessor` class loads text data, tokenizes it, and prepares it for training.
2. **Model Training**: The `ModelTrainer` class trains a language model on the processed data.
3. **Model Evaluation**: The `ModelEvaluator` class evaluates the trained model using metrics like perplexity.
4. **Text Generation**: The `TextGenerator` class uses the trained model to generate new text.

## Extending the Project

You can extend this project in several ways:

- Implement more sophisticated tokenization (e.g., subword tokenization)
- Add support for different model architectures (e.g., Transformer-based models)
- Implement more advanced training techniques (e.g., curriculum learning)
- Add support for different languages or domains

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# LLMPromptCS
