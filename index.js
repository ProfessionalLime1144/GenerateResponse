import express from "express";
import { config } from "dotenv";
import OpenAI from "openai";
import { CharacterTextSplitter, RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import PdfParse from "./modules/pdfParsed.js";

let initializeClient;
config();

const app = express();

app.listen(3000, () => {
  console.log("Connected to port 3000.");
  initializeClient = new OpenAIClient();
});

app.get("/", async(req, res)) {
  let response = await initializeClient.generateResponse("https://ocw.mit.edu/ans7870/9/9.00SC/MIT9_00SCF11_text.pdf", "Explain psychology");
  res.send(response);
});

class OpenAIClient {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
  }
  
  async generateResponse(url, question) {
    let startTime = Date.now();

    // Fetch the PDF file as binary data
    try {
      // const url = "https://ocw.mit.edu/ans7870/9/9.00SC/MIT9_00SCF11_text.pdf";
      const response = await fetch(url);
      if (!response.ok) {
        console.log(url);
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const data = await response.arrayBuffer();

      // Convert binary data to text and then embed it
      const parsedData = await PdfParse(data);

      const text = parsedData.text;
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100
      });

      const docs = await splitter.createDocuments([text]);
      let arrayDocs = [];

      for (const doc of docs) {
        arrayDocs.push(doc.pageContent);
      }

      arrayDocs = arrayDocs.slice(0, 1500);


      // Send a POST request to the Hugging Face API
      const hfResponse = await fetch('https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer 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'
        },
        body: JSON.stringify({
          inputs: {
            "source_sentence": question,
            "sentences": arrayDocs
          }
        })
      });

      if (!hfResponse.ok) {
        throw new Error(`HTTP error! Status: ${hfResponse.status}`);
      }

      const hfData = await hfResponse.json();

      let objectEmbeddings = [];

      for (let i = 0; i < hfData.length; i++) {
        objectEmbeddings.push({
          chunk: arrayDocs[i],
          embedding: hfData[i]
        });
      }

      let arrayEmbeddings = objectEmbeddings.sort((a, b) => b.embedding - a.embedding).slice(0, 3);
      let chunkedEmbeddings = arrayEmbeddings.map(obj => obj.chunk);
      chunkedEmbeddings = chunkedEmbeddings.join('');

      // Generate gpt 3.5 response
      const chatResponse = await this.openai.chat.completions.create({
        messages: [
          { role: "user", content: chunkedEmbeddings },
          { role: "user", content: question }
        ],
        model: "gpt-3.5-turbo",
      });
      let responseText = chatResponse.choices[0].message.content;

      // Print the number of tokens used
      const tokensUsed = JSON.stringify(chatResponse.usage);
      console.log(`Tokens used: ${tokensUsed}`);
      console.log("Time to retrieve response: " + ((Date.now() - startTime) / 1000) + " seconds.\n");

      return responseText;
    } catch (error) {
      console.error('Error:', error);
    }
  }
}
