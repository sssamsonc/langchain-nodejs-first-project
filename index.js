import dotenv from 'dotenv';
import {z} from 'zod';
import {OpenAI} from "@langchain/openai";
import {PromptTemplate} from "@langchain/core/prompts";
import {StructuredOutputParser} from "@langchain/core/output_parsers";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY not found!!");
}

const parser = StructuredOutputParser.fromZodSchema(
    z.object({
        name: z.string().describe("his name"),
        age: z.number().describe("his age in valid number format, eg: 30"),
        bio: z.string().describe("his short bio"),
        education: z.string().optional().describe("the university or school he graduated from"),
        interests: z.array(z.string()).describe("his interests"),
    })
);

const formatInstructions = parser.getFormatInstructions();

const promptTemplate = PromptTemplate.fromTemplate(
    `Generate information about a person\n{description}\n{format_instructions}`
);

const model = new OpenAI({
    model: "gpt-3.5-turbo-instruct",
    temperature: 0,
    maxTokens: -1,
    //timeout: undefined,
    maxRetries: 2,
    apiKey: process.env.OPENAI_API_KEY,
    callbacks: [
        {
            handleLLMEnd(output) {
                console.log(JSON.stringify(output, null, 2));
            },
        },
    ],
});

async function generateCharacter() {
    try {
        const prompt = await promptTemplate.format({
            format_instructions: formatInstructions,
            description: "A software engineer in Canada, born in Hong Kong.",
        });

        console.log("prompt::", prompt)

        const response = await model.invoke(prompt);
        console.log("response::", response)
        const parsedOutput = await parser.parse(response);

        console.log("parsedOutput::", parsedOutput);
    } catch (error) {
        console.error(error);
    }
}

generateCharacter();