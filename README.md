# Retrieval-Augmented Generation (RAG) Application

This repository contains a Node.js-based Retrieval-Augmented Generation (RAG) application leveraging the LangChain library. The app demonstrates the integration of PDF documents and web content into a question-answering system using memory-based vector storage and embeddings from Hugging Face.

## Features

- **PDF Document Integration**: Load and process PDF files for knowledge retrieval.
- **Web Content Integration**: Scrape and process web pages for real-time knowledge retrieval.
- **Vector Storage**: Use memory-based vector storage for efficient document retrieval.
- **Embeddings**: Generate embeddings using Hugging Face's inference APIs.
- **Customizable Chains**: Utilize LangChain’s chains for history-aware retrieval and document combination.
- **Interactive Chat Interface**: Simple command-line interface to query the system.

---

## Installation

### Prerequisites

- Node.js (v18 or later)
- npm (Node Package Manager)
- Hugging Face API key (for embeddings)
- Environment variables configured in a `.env` file:
  ```
  GROQ_API_KEY=your_groq_cloud_api_key
  LANGCHAIN_API_KEY=your_langchain_api_key
  HUGGINGFACEHUB_API_KEY=your_huggingface_api_key
  ```

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/duttasumit/NodeRAG.git
   cd NodeRAG
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Add a `.env` file with the necessary API key:
   ```
   GROQ_API_KEY=your_groq_cloud_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   HUGGINGFACEHUB_API_KEY=your_huggingface_api_key
   ```

4. Run the application:
   ```
   node index.js
   node pdfLoader.js
   node webLoader.js
   ```

---

## Usage

### Directory Structure

- `index.js`: Basic single query app with web scrapped content.
- `pdfLoader.js`: Handles loading and processing PDF documents.
- `webLoader.js`: Handles scraping and processing web content.

### Loading PDFs

The `pdfLoader.js` file uses `DirectoryLoader` and `PDFLoader` from LangChain to load and split PDFs for vector storage:
```javascript
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
```
Run the loader with:
```bash
node pdfLoader.js
```

### Scraping Web Content

The `webLoader.js` file uses `CheerioWebBaseLoader` to scrape web pages for content retrieval:
```javascript
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
```
Run the loader with:
```bash
node webLoader.js
```

### Querying

Interact with the system through the command-line interface to ask questions based on the loaded knowledge base.

---

## Packages Used

- **LangChain**: Core library for document loaders, text splitters, and chains.
- **@langchain/community**: Community-contributed loaders and embeddings.
- **dotenv**: Environment variable management.
- **readline-sync**: For interactive command-line inputs.

---

## Examples

### Sample Query
1. Query the system from PDF content:
   ```bash
   node pdfLoader.js
   ```
   Example query:
   ```
   > What is the benefits for mid-level employees?
   > What are the compensations?
   ```

2. Query the system from Web Content:
   ```bash
   node webLoader.js
   ```
   Example query:
   ```
   > What is Task Decomposition?
   > What are the common ways of doin it?
   ```

---

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- [LangChain](https://js.langchain.com/v0.2/docs/introduction/)
- [Hugging Face](https://huggingface.co/) for embeddings API.
- [Groq Cloud](https://console.groq.com/) for llm model.

