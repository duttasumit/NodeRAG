/* FOR RAG, we need:
1. Some type of loader to load the data (pdfloader, webloader)
2. Text splitter to split the text loaded from the loader (RecursiveCharaterTextSplitter)
3. Embeddings to convert text into embeddings (OpenAIEmbeddings, OllamaEmbeddings)
4. Vector DB to load the text embeddings (FAISS or Memory)
5. Retrieval chain to recieve the documents and pass to the stuffdocument chain
6. StuffDocument Chain to send list of documents to the model
7. chatprompttemplate for question-answering chain.
8. Messageplaceholder variable chat_history to manage history
9. createhistoryawareretriever to manage when chat_history is empty, else chain
10. Chatmessagehistory to manage session wise history
11. RunnableWithMessageHistory handles injecting chat history into inputs
*/


import { ChatGroq } from "@langchain/groq";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts"
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import "dotenv/config";


// 1. Initialize the llm model.
const llm = new ChatGroq({
    model: 'mixtral-8x7b-32768',
    temperature: 0,
    apiKey: process.env.GROQ_API_KEY
})


// 2. Initialize the loader and load the content.
const loader = new CheerioWebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    {
        selector: ".post-content, .post-title, .post-header",
    }
)
const docs = await loader.load()


// 3. Initialize the textsplitter and split the content (docs).
const textsplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
})
const splits = await textsplitter.splitDocuments(docs)


// 4. Store the splits in the memoryvectorstore with embeddings.
const hfembeddings = new HuggingFaceInferenceEmbeddings({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    apiKey: process.env.HUGGINGFACEHUB_API_KEY
})
const store = await MemoryVectorStore.fromDocuments(splits, hfembeddings)
const retriever = store.asRetriever()


// 5. Contextualize the Question, use Chatprompttemplate.
const contextualizeQSystemPrompt =
  "Given a chat history and the latest user question " +
  "which might reference context in the chat history, " +
  "formulate a standalone question which can be understood " +
  "without the chat history. Do NOT answer the question, " +
  "just reformulate it if needed and otherwise return it as is.";

const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"]
])


// 6. Use createHistoryRetriever to manage contexts and chaining.
const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: contextualizeQPrompt
})


// 7. Give system prompt to Answer Question.
const systemPrompt =
  "You are an assistant for question-answering tasks. " +
  "Use the following pieces of retrieved context to answer " +
  "the question. If you don't know the answer, say that you " +
  "don't know. Use three sentences maximum and keep the " +
  "answer concise." +
  "\n\n" +
  "{context}";

const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"]
])


// 8. Call createstuffdocumentchain to Pass the list of documents to the model.
const qnachain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt
})


// 9. Call createRetrievalChain to pass the documents to the stuffdocumentchain.
const ragchain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: qnachain
})


// 10. Manage chat history statefully.
const sessionStore = {}

function getSessionHistory(sessionId) {
    if(!(sessionId in sessionStore)) {
        sessionStore[sessionId] = new ChatMessageHistory()
    }
    return sessionStore[sessionId]
}

// Initialize RunnableWithMessageHistory to handle chat history.
const ragchainConversation = new RunnableWithMessageHistory({
    runnable: ragchain,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    outputMessagesKey: "answer"
})


// 11. Use the model to query.
const response = await ragchainConversation.invoke(
    { input: "What is Task Decomposition?"},
    { configurable: { sessionId: "abc123" }}
)

console.log(response.answer)