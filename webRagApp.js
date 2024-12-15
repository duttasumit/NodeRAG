/* For RAG, we need:
1. Chat model - llama3-8b-8192 from Groq cloud.
2. CheerioWebLoader - load content from web.
3. recursiveCharacterTextSplitter - to split texts to store in vector store.
4. MemoryVectorStore - Vector store.
5. HuggingfaceInferenceEmbeddings - Embeddings.
6. ChatPromptTemplate - Contextual and QnA prompts.
7. MessagesPlaceholder - "chat_history" for past message store
8. createHistoryAwareRetriever - for managing chat histories.
9. createStuffDocumentChain - send list of documents to the model.
10. createRetrievalChain - retrieve and pass documents to the stuffdocumentchain.
11. ChatMessageHistory - to create session wise chat.
12. RunnableWithMessageHistory - inject chat history to inputs.
*/

import { ChatGroq } from "@langchain/groq";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import 'dotenv/config';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const readlineSync = require('readline-sync');


const llm = new ChatGroq({
    model: 'llama3-8b-8192',
    temperature: 0
})

const sessionStore = {}

function getSessionHistory(sessionId) {
    if(!(sessionId in sessionStore)) {
        sessionStore[sessionId] = new ChatMessageHistory()
    }
    return sessionStore[sessionId]
}

async function main() {
    console.log('Initializing RAG Application....')

    const loader = new CheerioWebBaseLoader(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        {
            selector: ".post-content, .post-title, .post-header",
        }
    )
    const docs = await loader.load()
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200
    })
    const splits = await splitter.splitDocuments(docs)
    
    const hfembeddings = new HuggingFaceInferenceEmbeddings({
        model: 'sentence-transformers/all-MiniLM-L6-v2',
    })
    const store = await MemoryVectorStore.fromDocuments(splits, hfembeddings)
    const retriever = store.asRetriever()

    console.log('RAG Application is ready. Start asking questions.')

    while (true) {
        const userQuery = readlineSync.question('\nYou: ')
    
        if (userQuery.toLowerCase() === 'exit') {
            console.log('Goodbye!')
            break;
        }

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
        
        const historyAwareRetriever = await createHistoryAwareRetriever({
            llm,
            retriever,
            rephrasePrompt: contextualizeQPrompt
        })
        
        
        const systemPrompt =
        "You are an assistant for question-answering tasks. " +
        "Use the following pieces of retrieved context to answer " +
        "the question. If you don't know the answer, say that you " +
        "don't know. Use three sentences maximum and keep the " +
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
        
        const ragchainConversation = new RunnableWithMessageHistory({
            runnable: ragchain,
            getMessageHistory: getSessionHistory,
            inputMessagesKey: "input",
            historyMessagesKey: "chat_history",
            outputMessagesKey: "answer"
        })
        
        const response = await ragchainConversation.invoke(
            {input: userQuery},
            {configurable: { sessionId: 'abc123'}}
        )

        console.log(`AI: ${response.answer}`)
    }
}


main().catch(console.error)