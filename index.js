import { ChatGroq } from "@langchain/groq";
import 'dotenv/config'

const GROQ_API_KEY = process.env.GROQ_API_KEY

const llm = new ChatGroq({
    model: 'mixtral-8x7b-32768',
    temperature: 0,
    apiKey: GROQ_API_KEY
})

/* FOR RAG, we need:
1. Some type of loader to load the data (pdfloader, webloader)
2. Text splitter to split the text loaded from the loader (RecursiveCharaterTextSplitter)
3. Embeddings to convert text into embeddings (OpenAIEmbeddings, OllamaEmbeddings)
4. Vector DB to load the text embeddings (FAISS or Memory)
5. Retrieval chain
6. StuffDocument Chain
7. chatprompttemplate for question-answering chain.
*/

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { createRetrievalChain } from 'langchain/chains/retrieval'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

// Loading from web and create doucments.
const webloader = new CheerioWebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    {
        selector: ".post-content, .post-title, .post-header",
    }
)
const docs = await webloader.load()


// Splitting docs.
const textsplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 50
})
const splits = await textsplitter.splitDocuments(docs)


// Creating Embeddings.
const embeddings = new HuggingFaceInferenceEmbeddings({
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
});

// Store in vectorestore.
const vectorestore = await MemoryVectorStore.fromDocuments(
    splits,
    embeddings
)
const retriever = vectorestore.asRetriever()


// Use retriever for QnA Chain.
const systemPrompt =
  "You are an assistant for question-answering tasks. " +
  "Use the following pieces of retrieved context to answer " +
  "the question. If you don't know the answer, say that you " +
  "don't know. Use three sentences maximum and keep the " +
  "answer concise." +
  "\n\n" +
  "{context}";


import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever'
import { MessagesPlaceholder } from '@langchain/core/prompts'

const contextualizeQSystemPrompt =
  "Given a chat history and the latest user question " +
  "which might reference context in the chat history, " +
  "formulate a standalone question which can be understood " +
  "without the chat history. Do NOT answer the question, " +
  "just reformulate it if needed and otherwise return it as is.";


const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ["human", "{input}"] 
])

const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever,
    rephrasePrompt: contextualizeQPrompt
})


const qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder('chat_history'),
    ["human", "{input}"]
])

const questionAnswerChain2 = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt
})

const ragchain2 = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: questionAnswerChain2
})


import { HumanMessage, AIMessage } from '@langchain/core/messages'

let chatHistory = []

const question = "What is Task Decomposition?"
const aiMsg1 = await ragchain2.invoke({
    input: question,
    chat_history: chatHistory
})

chatHistory = chatHistory.concat([
    new HumanMessage(question),
    new AIMessage(aiMsg1.answer)
])

const secondQuestion = "What are common ways of doing it?"
const aiMsg2 = await ragchain2.invoke({
    input: secondQuestion,
    chat_history: chatHistory
})

console.log(aiMsg2.answer)