# RAG PDF Summarizer & Q&A

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) pipeline to facilitate question-answering based on the content of a PDF document. The system leverages the LangChain framework to orchestrate the workflow, from document loading and processing to generating context-aware answers. It utilizes Hugging Face models for creating text embeddings and for the final text generation, ensuring high-quality, relevant responses derived directly from the provided document.

The primary goal of this project is to provide a clear and functional example of a RAG system, enabling users to query their own PDF files and receive accurate answers. It is designed to be both a practical tool and an educational resource for understanding the components of a modern RAG pipeline.

## Features

- **PDF Document Loading**: Directly loads and processes text from PDF files.
- **Efficient Text Splitting**: Employs a recursive character splitter to break down documents into manageable chunks for analysis.
- **Vector Embeddings**: Uses Hugging Face embedding models to convert text chunks into vector representations.
- **In-Memory Vector Storage**: Leverages FAISS (Facebook AI Similarity Search) for efficient similarity searches on the text vectors.
- **Context-Aware Q&A**: Answers user questions based on the contextual information retrieved from the document.
- **Powered by Hugging Face**: Integrates with the Hugging Face ecosystem for both state-of-the-art embedding and language models.

## Installation

Follow these steps to set up the project locally.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Crashlar/rag-pdf-summarizer.git
    cd rag-pdf-summarizer
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    This project requires a Hugging Face API token to access the language model.

    - Create a file named `.env` in the root of the project directory.
    - Add your Hugging Face API token to the file as follows:
      ```
      HUGGINGFACEHUB_API_TOKEN="your_hf_token_here"
      ```

## Usage

1.  **Add Your PDF**
    Place the PDF file you want to query into the `data/` directory.

2.  **Configure the Script**
    Open `src/main.py` and modify the following variables:
    - Update `pdf_path` to point to your PDF file within the `data` directory.
    - Change the `question` variable inside the `PdfRag` function to ask what you want to know.

    ```python
    # In src/main.py

    # ...

    if __name__ == "__main__":
        # ...
        # Update this path to your PDF
        pdf_path = data_dir / "your_document_name.pdf" 
        
        # ...

    # Inside the PdfRag function...
    
    # Update this question
    question = "What is the main topic of the document?"
    
    # ...
    ```

3.  **Run the Script**
    Execute the main script from the root of the project directory:
    ```bash
    python src/main.py
    ```
    The script will process the PDF and print the answer to your question in the console.

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to report a bug, please feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a pull request.

## License

This project is licensed under the **MIT License**.

This license grants you the freedom to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software. It is a permissive license that allows for reuse within both open-source and proprietary software.

The only significant obligation is that the original copyright and license notice must be included in any substantial portion of the software.

For the full legal text and details, please review the [LICENSE](LICENSE) file in the repository.

## Contact

For any questions or support, please open an issue on the GitHub repository.
