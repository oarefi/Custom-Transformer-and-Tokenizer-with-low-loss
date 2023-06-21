# Machine Translation with Transformers

Welcome to the Machine Translation with Transformers project! This project demonstrates the implementation of a Transformer model for machine translation tasks. Follow the steps below to use the project:

## Usage

1. **Prepare your training data:** You need to provide a CSV file containing the source language sentences and the corresponding target language sentences. The CSV file should have two columns: "source" and "target". Each row represents a sentence pair.

2. **Initialize the tokenizer:** The tokenizer is responsible for tokenizing the sentences and converting them into numerical representations. To initialize the tokenizer, create an instance of the Tokenizer class and load the vocabulary dictionary from a JSON file using the `load_word2idx` method.

3. **Tokenize the sentences:** Use the `tokenize` method of the tokenizer to convert the sentences into tokenized form. This step is necessary to prepare the data for training the model.

4. **Prepare the input tensors:** Pad or truncate the tokenized sentences to a fixed length using the `max_seq_length` variable. This ensures that all sentences have the same length. Convert the tokenized sentences into tensors using `torch.tensor`.

5. **Define the Transformer model:** Instantiate an instance of the Transformer class and specify the required parameters such as source and target vocabulary size, model dimensions, number of attention heads, number of layers, and dropout rate.

6. **Train the model:** Use the prepared input tensors as the input to the `forward` method of the Transformer model. The model will perform the forward pass and return the output predictions.

7. **Translate sentences:** Use the `translate_sentence` method of the Transformer model to translate a given sentence from the source language to the target language. This method takes a sentence as input, tokenizes it, performs the translation, and returns the translated sentence.

For a complete example and usage details, please refer to the [Documentation](docs/README.md) folder.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
