import dotenv from 'dotenv';
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import {OpenAIEmbeddings} from "@langchain/openai";
import {ChatOpenAI} from "@langchain/openai";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";
import {RunnableLambda, RunnableMap, RunnableSequence} from "@langchain/core/runnables";
import {formatDocumentsAsString} from "langchain/util/document";
import {ChatPromptTemplate} from "@langchain/core/prompts";
import * as readline from "node:readline";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY not found!!");
}

const loadPDFs = async (path) => {
    const loader = new DirectoryLoader(path, {
        ".pdf": (path) => new PDFLoader(path, {
            parsedItemSeparator: "",
        }),
    });
    return await loader.load();
};

const splitDocs = async (docs) => {
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 700,
        chunkOverlap: 100,
    });
    return await splitter.splitDocuments(docs);
};

const storeInVectorDB = async (docs) => {
    const embeddings = new OpenAIEmbeddings({
        model: "text-embedding-3-large"
    });
    return await MemoryVectorStore.fromDocuments(docs, embeddings);
};

const createChain = async (vectorStore) => {
    const retriever = vectorStore.asRetriever();
    const model = new ChatOpenAI({
        model: "gpt-3.5-turbo",
        temperature: 0,
        callbacks: [
            {
                handleLLMEnd(output) {
                    console.log(JSON.stringify(output, null, 2));
                },
            },
        ],
    });

    const chatPrompt = ChatPromptTemplate.fromMessages([
        ["system", `You are a helpful assistant. You must only use the context provided. If the answer is not found in the context, respond exactly: "I don’t know based on the document."`],
        ["human", `Context:\n{context}\n\nQuestion:\n{question}`],
    ]);

    return RunnableSequence.from([
        // {
        //     context: async (input) => {
        //         const retrievedDocs = await retriever.invoke(input);
        //         return formatDocumentsAsString(retrievedDocs);
        //     },
        //     question: (input) => input,
        // },
        // async ({context, question}) =>
        //     model.invoke(`Answer the question based on the context.\n\nContext:\n${context}\n\nQuestion: ${question}`),

        RunnableMap.from({
            context: async (input) => {
                const retrievedDocs = await retriever.invoke(input);
                return formatDocumentsAsString(retrievedDocs);
            },
            question: (input) => input
        }),
        // RunnableLambda.from(({context, question}) => {
        //     const prompt = `Answer this:\nContext: ${context}\nQ: ${question}`
        //     console.log("prompt::", prompt);
        //     return prompt;
        // }),
        //chatPrompt,
        RunnableLambda.from(async (input) => { //debug and use chatPrompt,
            const messages = await chatPrompt.formatMessages(input);
            console.log("prompt::", messages);
            return messages;
        }),
        model,
    ]);
};

const run = async () => {
    const docs = await loadPDFs("./pdfs/");
    const splitDocsResult = await splitDocs(docs);
    const vectorStore = await storeInVectorDB(splitDocsResult);
    const chain = await createChain(vectorStore);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    const ask = () => {
        rl.question("Ask me! (or type 'exit'): ", async (input) => {
            if (input.toLowerCase() === "exit") {
                rl.close();
                return;
            }

            try {
                const response = await chain.invoke(input); //question
                console.log("response::", response.content || response);
            } catch (err) {
                console.error("error::", err);
            }
            ask(); // repeat
        });
    };

    console.log("PDFs loaded! Ask me now.");
    ask();
}

await run();
