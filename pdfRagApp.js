/* For PDF RAG App:
1. Chat model from Groq.
2. DirectoryLoader - to load directory with pdf files
3. PDFLoader to load pdf files
4. pdf-parser to parse from pdf
5. RecursiveCharacterTextSplitter
6. HugginFaceInferenceEmbeddings
7. MemoryVectorStore
8. createHistoryAwareRetriever
8. ChatPromptTemplate
9. MessagesPlaceholder
9. createStuffDocumentChain
10. createRetrievalChain
11. ChatMessageHistory
12. RunnableWithMessageHistory
*/

import { ChatGroq } from "@langchain/groq";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import "dotenv/config";

import { createRequire } from "module";
const require = createRequire(import.meta.url);
const readlineSync = require('readline-sync');


const llm = new ChatGroq({
    model: 'llama3-8b-8192',
    temperature: 0
})

const sessionStore = {}

async function getSessionHistory(sessionId) {
    if(!(sessionId in sessionStore)) {
        sessionStore[sessionId] = new ChatMessageHistory()
    }
    return sessionStore[sessionId]
}


async function main() {
    console.log("Initializing RAG Application...........")

    const directoryLoader = new DirectoryLoader(
        "../NodeRAG/knowledgebase",
        {
            ".pdf": (path) => new PDFLoader(path),
        }
    )
    const directoryDocs = await directoryLoader.load()
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    const docs = await splitter.splitDocuments(directoryDocs)
    
    const hfembeddings = new HuggingFaceInferenceEmbeddings({
        model: "sentence-transformers/all-MiniLM-L6-v2",
    })
    
    const store = await MemoryVectorStore.fromDocuments(docs, hfembeddings)
    const retriever = store.asRetriever()

    console.log("Welcome to AishMit LLC. HR Assistant. Start Asking Questions.")

    while(true) {
        const userQuery = readlineSync.question("\nYou: ")
        if(userQuery.toLowerCase() === "exit") {
            console.log("Goodbye!")
            break;
        }
    
        const contextualizeQSystemPrompt =
        "Given a chat history and the latest user question " +
        "which might reference context in the chat history, " +
        "formulate a standalone question which can be understood " +
        "without the chat history. Do NOT answer the question, " +
        "just reformulate it if needed and otherwise return it as is.";

        const contextualizeQPrompt = await ChatPromptTemplate.fromMessages([
            ["system", contextualizeQSystemPrompt],
            new MessagesPlaceholder("chat_history"),
            ["human", "{input}"]
        ])
        const historyAwareRetriever = await createHistoryAwareRetriever({
            llm,
            retriever,
            rephrasePrompt: contextualizeQPrompt
        })

        const systemPrompt =
        "You are an HR assistant for question-answering tasks from AishMit LLC. " +
        "Use the following pieces of retrieved context to answer " +
        "the question. If you don't know the answer, say that you " +
        "don't know. Before answering, ask user the country if you don't know " +
        "the user country. Don't assume by yourself. Answer " +
        "based on that country only if the country is India or Australia " +
        "else say that you don't know. Use three sentences maximum and keep the " +
        "answer concise." +
        "\n\n" +
        "{context}";

        const qaPrompt = await ChatPromptTemplate.fromMessages([
            ["system", systemPrompt],
            new MessagesPlaceholder("chat_history"),
            ["human", "{input}"]
        ])
        const qnachain = await createStuffDocumentsChain({
            llm,
            prompt: qaPrompt
        })
        const ragchain = await createRetrievalChain({
            retriever: historyAwareRetriever,
            combineDocsChain: qnachain
        })

        const ragqnachain = new RunnableWithMessageHistory({
            runnable: ragchain,
            getMessageHistory: getSessionHistory,
            inputMessagesKey: "input",
            historyMessagesKey: "chat_history",
            outputMessagesKey: "answer"
        })

        const response = await ragqnachain.invoke(
            { input: userQuery },
            { configurable: { sessionId: 'abc123' }}
        )

        console.log(`AI: ${response.answer}`)
    }

}


main().catch(console.error)