import http from "http";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

import {
  loadModel,
  completion,
  unloadModel,
  LLAMA_3_2_1B_INST_Q4_0
} from "@qvac/sdk";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HOST = "127.0.0.1";
const PORT = 4028;

const PUBLIC_DIR = path.join(__dirname, "public");
const MAX_FILE_SIZE = 2 * 1024 * 1024;

const ALLOWED_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".json",
  ".csv"
]);

const documents = new Map();

let modelId = null;
let modelLoading = false;
let modelProgress = 0;
let modelStatus = "Model not loaded";

/* =========================================================
   RESPONSE HELPERS
   ========================================================= */

function sendJson(res, statusCode, data) {
  const body = JSON.stringify(data);

  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body)
  });

  res.end(body);
}

function sendText(
  res,
  statusCode,
  text,
  contentType = "text/plain; charset=utf-8"
) {
  const body =
    Buffer.isBuffer(text)
      ? text
      : Buffer.from(String(text));

  res.writeHead(statusCode, {
    "Content-Type": contentType,
    "Content-Length": body.length
  });

  res.end(body);
}

/* =========================================================
   TEXT CLEANING
   ========================================================= */

function cleanText(value) {
  return String(value ?? "")
    .replace(/\u0000/g, "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function normalizeForComparison(value) {
  return cleanText(value)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/* =========================================================
   AI OUTPUT CLEANING
   ========================================================= */

function collapseRepeatedWords(text) {
  let output = String(text || "");

  for (let i = 0; i < 10; i++) {
    output = output.replace(
      /\b([A-Za-z][A-Za-z'-]{1,30})\s+\1\b/gi,
      "$1"
    );

    output = output.replace(
      /\b(\d{1,5})\1\b/g,
      "$1"
    );

    output = output.replace(
      /\b((?:[A-Za-z0-9][A-Za-z0-9'-]*\s+){1,8}[A-Za-z0-9][A-Za-z0-9'-]*)\s+\1\b/gi,
      "$1"
    );
  }

  return output;
}

function removeRepeatedLines(text) {
  const lines = String(text || "")
    .split(/\n+/)
    .map(line => line.trim())
    .filter(Boolean);

  const result = [];
  const seen = new Set();

  for (const line of lines) {
    const key = normalizeForComparison(line);

    if (!key) {
      continue;
    }

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    result.push(line);
  }

  return result.join("\n");
}

function cleanAIAnswer(text) {
  let output = cleanText(text);

  output = collapseRepeatedWords(output);
  output = removeRepeatedLines(output);

  output = output
    .replace(/\s+([,.;!?])/g, "$1")
    .replace(/([.!?])\s*\1+/g, "$1")
    .replace(/:\s*:/g, ":")
    .replace(/-\s*-/g, "-");

  return output.trim();
}

/* =========================================================
   SENTENCES AND CHUNKS
   ========================================================= */

function splitIntoSentences(text) {
  const cleaned = cleanText(text);

  if (!cleaned) {
    return [];
  }

  const matches =
    cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g);

  if (!matches) {
    return [cleaned];
  }

  return matches
    .map(sentence => sentence.trim())
    .filter(Boolean);
}

function splitIntoChunks(text, maxLength = 900) {
  const cleaned = cleanText(text);

  if (!cleaned) {
    return [];
  }

  const paragraphs = cleaned
    .split(/\n\s*\n/)
    .map(part => part.trim())
    .filter(Boolean);

  const chunks = [];

  for (const paragraph of paragraphs) {
    if (paragraph.length <= maxLength) {
      chunks.push(paragraph);
      continue;
    }

    const sentences = splitIntoSentences(paragraph);

    let current = "";

    for (const sentence of sentences) {
      if (!current) {
        current = sentence;
        continue;
      }

      if (
        (current + " " + sentence).length <= maxLength
      ) {
        current += " " + sentence;
      } else {
        chunks.push(current.trim());
        current = sentence;
      }
    }

    if (current) {
      chunks.push(current.trim());
    }
  }

  if (chunks.length === 0) {
    chunks.push(cleaned.slice(0, maxLength));
  }

  return chunks;
}

/* =========================================================
   DOCUMENT SEARCH
   ========================================================= */

function tokenize(text) {
  return normalizeForComparison(text)
    .split(/\s+/)
    .filter(word => word.length >= 3);
}

function calculateRelevance(question, chunk) {
  const questionTokens = [
    ...new Set(tokenize(question))
  ];

  const chunkText = normalizeForComparison(chunk);

  if (questionTokens.length === 0) {
    return 0;
  }

  let matches = 0;

  for (const token of questionTokens) {
    if (chunkText.includes(token)) {
      matches++;
    }
  }

  const score =
    matches / questionTokens.length;

  return Math.min(
    1,
    Math.max(0, score)
  );
}

function searchDocuments(question, limit = 8) {
  const results = [];

  for (const doc of documents.values()) {
    for (const chunk of doc.chunks) {
      const relevance =
        calculateRelevance(
          question,
          chunk.text
        );

      results.push({
        documentId: doc.id,
        documentName: doc.name,
        chunkId: chunk.id,
        text: chunk.text,

        relevanceScore: relevance,
        score: relevance,
        relevance: relevance,
        rankScore: relevance
      });
    }
  }

  results.sort((a, b) => {
    if (
      b.relevanceScore !==
      a.relevanceScore
    ) {
      return (
        b.relevanceScore -
        a.relevanceScore
      );
    }

    return a.documentName.localeCompare(
      b.documentName
    );
  });

  return results.slice(0, limit);
}

function getAllEvidence() {
  const results = [];

  for (const doc of documents.values()) {
    for (const chunk of doc.chunks) {
      results.push({
        documentId: doc.id,
        documentName: doc.name,
        chunkId: chunk.id,
        text: chunk.text,

        relevanceScore: 1,
        score: 1,
        relevance: 1,
        rankScore: 1
      });
    }
  }

  return results;
}

/* =========================================================
   QVAC MODEL
   ========================================================= */

async function ensureModel() {
  if (modelId) {
    return modelId;
  }

  if (modelLoading) {
    while (modelLoading) {
      await new Promise(resolve =>
        setTimeout(resolve, 150)
      );
    }

    if (modelId) {
      return modelId;
    }

    throw new Error(
      "QVAC model failed to load."
    );
  }

  modelLoading = true;
  modelProgress = 0;
  modelStatus =
    "Loading QVAC model...";

  try {
    const id = await loadModel({
      modelSrc:
        LLAMA_3_2_1B_INST_Q4_0,

      modelType: "llm",

      onProgress(progress) {
        if (
          typeof progress === "number"
        ) {
          modelProgress =
            Math.max(
              0,
              Math.min(1, progress)
            );
        }

        modelStatus =
          "Loading QVAC model...";
      }
    });

    modelId = id;
    modelProgress = 1;
    modelStatus =
      "QVAC model ready";

    return id;
  } catch (error) {
    modelStatus =
      "Model loading failed";

    throw error;
  } finally {
    modelLoading = false;
  }
}

async function askQVAC(prompt) {
  const id =
    await ensureModel();

  const result = completion({
    modelId: id,

    history: [
      {
        role: "user",
        content: prompt
      }
    ],

    stream: true
  });

  let output = "";

  if (result?.tokenStream) {
    for await (
      const token of result.tokenStream
    ) {
      if (typeof token === "string") {
        output += token;
      } else if (token?.text) {
        output += token.text;
      } else if (token?.content) {
        output += token.content;
      }
    }
  }

  return cleanAIAnswer(output);
}

/* =========================================================
   PROMPT EVIDENCE
   ========================================================= */

function formatEvidenceForPrompt(
  sources
) {
  return sources
    .map((source, index) => {
      return [
        `[Source ${index + 1}] ${source.documentName}`,
        source.text
      ].join("\n");
    })
    .join("\n\n");
}

/* =========================================================
   AI OUTPUT VALIDATION
   ========================================================= */

function extractNumbers(text) {
  return (
    String(text || "").match(
      /\b\d+(?:\.\d+)?\b/g
    ) || []
  );
}

function containsUnsupportedNumbers(
  answer,
  evidence
) {
  const answerNumbers =
    extractNumbers(answer);

  const evidenceNumbers =
    extractNumbers(evidence);

  return answerNumbers.some(
    number =>
      !evidenceNumbers.includes(number)
  );
}

function hasHeavyRepetition(text) {
  const cleaned =
    normalizeForComparison(text);

  if (!cleaned) {
    return true;
  }

  const words =
    cleaned.split(/\s+/);

  if (words.length < 4) {
    return false;
  }

  let repeatedAdjacent = 0;

  for (
    let i = 1;
    i < words.length;
    i++
  ) {
    if (
      words[i] ===
      words[i - 1]
    ) {
      repeatedAdjacent++;
    }
  }

  if (repeatedAdjacent >= 2) {
    return true;
  }

  const phraseCounts =
    new Map();

  for (
    let size = 2;
    size <= 6;
    size++
  ) {
    for (
      let i = 0;
      i <= words.length - size;
      i++
    ) {
      const phrase =
        words
          .slice(i, i + size)
          .join(" ");

      const count =
        (phraseCounts.get(phrase) || 0) +
        1;

      phraseCounts.set(
        phrase,
        count
      );

      if (count >= 3) {
        return true;
      }
    }
  }

  return false;
}

function hasSuspiciousLength(text) {
  return cleanText(text).length > 3500;
}

function answerIsGrounded(
  answer,
  sources
) {
  const evidence =
    sources
      .map(source => source.text)
      .join("\n");

  if (
    !answer ||
    answer.length < 20
  ) {
    return false;
  }

  if (
    hasHeavyRepetition(answer)
  ) {
    return false;
  }

  if (
    hasSuspiciousLength(answer)
  ) {
    return false;
  }

  if (
    containsUnsupportedNumbers(
      answer,
      evidence
    )
  ) {
    return false;
  }

  return true;
}

/* =========================================================
   SAFE RESEARCH FALLBACK
   ========================================================= */

function evidenceFallback(
  question,
  sources
) {
  if (sources.length === 0) {
    return (
      "No evidence is available in the research library."
    );
  }

  const ranked =
    sources
      .map(source => ({
        source,
        score:
          calculateRelevance(
            question,
            source.text
          )
      }))
      .sort(
        (a, b) =>
          b.score - a.score
      );

  const selected =
    ranked
      .slice(
        0,
        Math.min(
          3,
          ranked.length
        )
      )
      .map(item => item.source);

  const sentences = [];

  for (const source of selected) {
    for (
      const sentence of
        splitIntoSentences(
          source.text
        )
    ) {
      if (
        sentences.length >= 3
      ) {
        break;
      }

      if (
        sentence.length >= 20
      ) {
        sentences.push({
          text: sentence,
          source:
            source.documentName
        });
      }
    }

    if (
      sentences.length >= 3
    ) {
      break;
    }
  }

  if (
    sentences.length === 0
  ) {
    return `Based on the supplied evidence, information was found in ${selected
      .map(
        source =>
          source.documentName
      )
      .join(", ")}.`;
  }

  const body =
    sentences
      .slice(0, 3)
      .map(
        item =>
          `${item.text} [${item.source}]`
      )
      .join(" ");

  return `Based on the supplied evidence, ${body}`;
}

/* =========================================================
   RESEARCH QUESTION
   ========================================================= */

async function researchQuestion(
  question
) {
  const sources =
    searchDocuments(
      question,
      8
    );

  if (sources.length === 0) {
    return {
      answer:
        "No evidence is available. Add research documents first.",
      sources: []
    };
  }

  const evidence =
    formatEvidenceForPrompt(
      sources
    );

  const prompt = `
You are a local evidence research assistant.

Answer the user's question using ONLY the supplied evidence.

USER QUESTION:
${question}

SUPPLIED EVIDENCE:
${evidence}

Rules:
- Do not invent facts.
- Do not invent people.
- Do not invent organizations.
- Do not invent dates.
- Do not invent numbers.
- Do not invent places.
- Every number in your answer must appear in the supplied evidence.
- Keep the answer concise.
- Do not repeat words.
- Do not repeat phrases.
- Do not repeat sentences.
- Do not add general knowledge.
- Do not use markdown tables.
- Answer in 2 to 4 clear sentences.
`;

  let aiAnswer = "";

  try {
    aiAnswer =
      await askQVAC(prompt);
  } catch (error) {
    console.error(
      "QVAC research error:",
      error.message
    );

    aiAnswer = "";
  }

  if (
    !answerIsGrounded(
      aiAnswer,
      sources
    )
  ) {
    aiAnswer =
      evidenceFallback(
        question,
        sources
      );
  }

  return {
    answer:
      cleanAIAnswer(
        aiAnswer
      ),

    sources
  };
}

/* =========================================================
   RESEARCH BRIEF
   ========================================================= */

async function researchBrief() {
  const docs =
    [...documents.values()];

  if (docs.length === 0) {
    return {
      result:
        "No research material is available."
    };
  }

  const allSentences = [];

  let sourceNumber = 1;

  for (const doc of docs) {
    const sentences =
      splitIntoSentences(
        doc.text
      );

    for (
      const sentence of sentences
    ) {
      if (
        sentence.trim()
      ) {
        allSentences.push({
          text:
            sentence.trim(),

          sourceNumber,

          documentName:
            doc.name
        });
      }
    }

    sourceNumber++;
  }

  const evidenceSentences =
    allSentences
      .filter(
        item =>
          item.text.length >= 20
      )
      .slice(0, 20);

  const overviewSentences =
    evidenceSentences
      .slice(0, 3);

  const issueSentences =
    evidenceSentences
      .filter(item =>
        /\b(?:concern|problem|polluted|pollution|waste|litter|lack|reported|issue)\b/i
          .test(item.text)
      )
      .slice(0, 5);

  const actionSentences =
    evidenceSentences
      .filter(item =>
        /\b(?:recommend|recommends|propose|proposed|proposal|install|cleanup|educational|activities|begin)\b/i
          .test(item.text)
      )
      .slice(0, 5);

  const result = [
    "RESEARCH BRIEF:",
    "",

    "OVERVIEW:",

    ...overviewSentences.map(
      item =>
        `- ${item.text} [Source ${item.sourceNumber}]`
    ),

    "",

    "KEY EVIDENCE:",

    ...evidenceSentences
      .slice(0, 8)
      .map(
        item =>
          `- ${item.text} [Source ${item.sourceNumber}]`
      ),

    "",

    "OBSERVED ISSUES:",

    ...(issueSentences.length
      ? issueSentences.map(
          item =>
            `- ${item.text} [Source ${item.sourceNumber}]`
        )
      : [
          "- No specific issue statements were identified."
        ]),

    "",

    "PROPOSED ACTIONS:",

    ...(actionSentences.length
      ? actionSentences.map(
          item =>
            `- ${item.text} [Source ${item.sourceNumber}]`
        )
      : [
          "- No specific proposed actions were identified."
        ])
  ].join("\n");

  return {
    result
  };
}

/* =========================================================
   FACT EXTRACTION
   ========================================================= */

function isFactSentence(
  sentence
) {
  const text =
    sentence.trim();

  if (!text) {
    return false;
  }

  const hasNumber =
    /\b\d+(?:\.\d+)?\b/
      .test(text);

  const hasDate =
    /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b/i
      .test(text) ||
    /\b20\d{2}\b/
      .test(text);

  const hasPlace =
    /\b(?:Creek|River|Riverside|community|area|location|district|barangay)\b/i
      .test(text);

  const hasClaim =
    /\b(?:reported|said|recommended|recommends|proposed|proposal|requested|collected|covered|conducted|expect|expects|contributed|approved)\b/i
      .test(text);

  return (
    hasNumber ||
    hasDate ||
    hasPlace ||
    hasClaim
  );
}

function deterministicFactExtraction() {
  const facts = [];

  let sourceNumber = 1;

  for (const doc of documents.values()) {
    const sentences =
      splitIntoSentences(
        doc.text
      );

    for (
      const sentence of sentences
    ) {
      if (
        !isFactSentence(
          sentence
        )
      ) {
        continue;
      }

      facts.push({
        sentence,
        sourceNumber,
        documentName:
          doc.name
      });
    }

    sourceNumber++;
  }

  if (facts.length === 0) {
    return [
      "FACT EXTRACTION:",
      "",
      "No structured facts were identified in the supplied documents."
    ].join("\n");
  }

  const unique = [];
  const seen = new Set();

  for (const fact of facts) {
    const key =
      normalizeForComparison(
        fact.sentence
      );

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    unique.push(fact);
  }

  const lines =
    unique
      .slice(0, 30)
      .map(
        fact =>
          `- ${fact.sentence} [Source ${fact.sourceNumber}]`
      );

  return [
    "FACT EXTRACTION:",
    "",
    ...lines
  ].join("\n");
}

/* =========================================================
   CONTRADICTION SCAN
   ========================================================= */

function extractNumericClaims() {
  const claims = [];

  let sourceNumber = 1;

  for (const doc of documents.values()) {
    const sentences =
      splitIntoSentences(
        doc.text
      );

    for (
      const sentence of sentences
    ) {
      const numbers =
        extractNumbers(
          sentence
        );

      if (
        numbers.length > 0
      ) {
        claims.push({
          sentence,
          numbers,
          sourceNumber,
          documentName:
            doc.name
        });
      }
    }

    sourceNumber++;
  }

  return claims;
}

function deterministicContradictionScan() {
  const claims =
    extractNumericClaims();

  const conflicts = [];

  for (
    let i = 0;
    i < claims.length;
    i++
  ) {
    for (
      let j = i + 1;
      j < claims.length;
      j++
    ) {
      const a = claims[i];
      const b = claims[j];

      if (
        a.sourceNumber ===
        b.sourceNumber
      ) {
        continue;
      }

      const aWords =
        new Set(
          tokenize(a.sentence)
        );

      const bWords =
        new Set(
          tokenize(b.sentence)
        );

      let overlap = 0;

      for (
        const word of aWords
      ) {
        if (
          bWords.has(word)
        ) {
          overlap++;
        }
      }

      const smaller =
        Math.max(
          1,
          Math.min(
            aWords.size,
            bWords.size
          )
        );

      const similarity =
        overlap / smaller;

      if (
        similarity >= 0.45 &&
        a.numbers.join(",") !==
          b.numbers.join(",")
      ) {
        conflicts.push({
          a,
          b
        });
      }
    }
  }

  const lines = [
    "CONTRADICTION SCAN:",
    ""
  ];

  if (
    conflicts.length === 0
  ) {
    lines.push(
      "No clear contradictions were found among the supplied documents."
    );
  } else {
    lines.push(
      "POSSIBLE CONFLICTS:"
    );

    for (
      const conflict of
        conflicts.slice(0, 10)
    ) {
      lines.push(
        `- [Source ${conflict.a.sourceNumber}] ${conflict.a.sentence}`
      );

      lines.push(
        `  [Source ${conflict.b.sourceNumber}] ${conflict.b.sentence}`
      );

      lines.push("");
    }

    lines.push(
      "These are possible conflicts detected from overlapping statements with different numeric values. Review the original documents before treating them as actual contradictions."
    );
  }

  return lines.join("\n");
}

/* =========================================================
   DOCUMENT API
   ========================================================= */

function getCounts() {
  let chunkCount = 0;

  for (
    const doc of documents.values()
  ) {
    chunkCount +=
      Array.isArray(doc.chunks)
        ? doc.chunks.length
        : 0;
  }

  return {
    documentCount:
      documents.size,

    chunkCount
  };
}

function publicDocument(doc) {
  return {
    id: doc.id,
    name: doc.name,
    size: doc.size,
    hash: doc.hash,
    createdAt: doc.createdAt,

    chunkCount:
      Array.isArray(doc.chunks)
        ? doc.chunks.length
        : 0,

    chunks:
      Array.isArray(doc.chunks)
        ? doc.chunks.map(
            chunk => ({
              id: chunk.id,
              size:
                chunk.text.length
            })
          )
        : []
  };
}

/* =========================================================
   REQUEST BODY
   ========================================================= */

async function readRequestBody(
  req,
  maxBytes =
    MAX_FILE_SIZE +
    1024 * 1024
) {
  const chunks = [];
  let total = 0;

  for await (
    const chunk of req
  ) {
    total += chunk.length;

    if (
      total > maxBytes
    ) {
      throw new Error(
        "Request body is too large."
      );
    }

    chunks.push(chunk);
  }

  return Buffer.concat(chunks);
}

/* =========================================================
   MULTIPART PARSER
   ========================================================= */

function parseMultipart(
  bodyBuffer,
  contentType
) {
  const match =
    contentType.match(
      /boundary=(?:"([^"]+)"|([^;]+))/i
    );

  if (!match) {
    throw new Error(
      "Multipart boundary is missing."
    );
  }

  const boundary =
    match[1] || match[2];

  const body =
    bodyBuffer.toString(
      "binary"
    );

  const delimiter =
    `--${boundary}`;

  const parts =
    body.split(
      delimiter
    );

  const files = [];

  for (
    const part of parts
  ) {
    if (
      !part ||
      part === "--\r\n" ||
      part === "--"
    ) {
      continue;
    }

    const separator =
      "\r\n\r\n";

    const headerEnd =
      part.indexOf(
        separator
      );

    if (
      headerEnd === -1
    ) {
      continue;
    }

    const headerText =
      part.slice(
        0,
        headerEnd
      );

    let content =
      part.slice(
        headerEnd +
          separator.length
      );

    if (
      content.endsWith(
        "\r\n"
      )
    ) {
      content =
        content.slice(
          0,
          -2
        );
    }

    if (
      content.endsWith(
        "--"
      )
    ) {
      content =
        content.slice(
          0,
          -2
        );
    }

    const filenameMatch =
      headerText.match(
        /filename="([^"]*)"/i
      );

    if (!filenameMatch) {
      continue;
    }

    const filename =
      filenameMatch[1];

    files.push({
      filename,

      buffer:
        Buffer.from(
          content,
          "binary"
        )
    });
  }

  return files;
}

/* =========================================================
   ADD DOCUMENT
   ========================================================= */

function addDocument(
  filename,
  buffer
) {
  const safeName =
    path.basename(filename);

  const extension =
    path.extname(
      safeName
    ).toLowerCase();

  if (
    !ALLOWED_EXTENSIONS.has(
      extension
    )
  ) {
    throw new Error(
      "Unsupported file type. Use TXT, MD, JSON or CSV."
    );
  }

  if (
    buffer.length >
    MAX_FILE_SIZE
  ) {
    throw new Error(
      "File is too large. Maximum size is 2 MB."
    );
  }

  const text =
    cleanText(
      buffer.toString(
        "utf8"
      )
    );

  if (!text) {
    throw new Error(
      "The uploaded document is empty."
    );
  }

  const hash =
    crypto
      .createHash("sha256")
      .update(buffer)
      .digest("hex");

  for (
    const existing of
      documents.values()
  ) {
    if (
      existing.hash === hash
    ) {
      throw new Error(
        "This document is already in the research library."
      );
    }
  }

  const id =
    crypto.randomUUID();

  const chunks =
    splitIntoChunks(
      text
    ).map(
      (chunkText, index) => ({
        id:
          `${id}-chunk-${index + 1}`,

        text: chunkText
      })
    );

  const document = {
    id,
    name: safeName,
    size: buffer.length,
    hash,

    createdAt:
      new Date().toISOString(),

    text,
    chunks
  };

  documents.set(
    id,
    document
  );

  return document;
}

/* =========================================================
   HTTP SERVER
   ========================================================= */

const server =
  http.createServer(
    async (req, res) => {
      try {
        const url =
          new URL(
            req.url,
            `http://${HOST}:${PORT}`
          );

        /* STATUS */

        if (
          req.method === "GET" &&
          url.pathname ===
            "/api/status"
        ) {
          const counts =
            getCounts();

          return sendJson(
            res,
            200,
            {
              online: true,

              modelLoaded:
                Boolean(modelId),

              modelLoading,

              progress:
                modelProgress,

              status:
                modelStatus,

              documentCount:
                counts.documentCount,

              chunkCount:
                counts.chunkCount,

              documents:
                counts.documentCount,

              chunks:
                counts.chunkCount,

              sdk: "0.19.0"
            }
          );
        }

        /* GET DOCUMENTS */

        if (
          req.method === "GET" &&
          url.pathname ===
            "/api/documents"
        ) {
          return sendJson(
            res,
            200,
            {
              documents:
                [
                  ...documents.values()
                ].map(
                  publicDocument
                )
            }
          );
        }

        /* LOAD MODEL */

        if (
          req.method === "POST" &&
          url.pathname ===
            "/api/model/load"
        ) {
          await ensureModel();

          return sendJson(
            res,
            200,
            {
              ok: true,

              modelLoaded: true,

              status:
                modelStatus
            }
          );
        }

        /* UPLOAD DOCUMENT */

        if (
          req.method === "POST" &&
          url.pathname ===
            "/api/documents"
        ) {
          const contentType =
            req.headers[
              "content-type"
            ] || "";

          if (
            !contentType
              .toLowerCase()
              .startsWith(
                "multipart/form-data"
              )
          ) {
            return sendJson(
              res,
              400,
              {
                error:
                  "Upload must use multipart/form-data."
              }
            );
          }

          const body =
            await readRequestBody(
              req
            );

          const files =
            parseMultipart(
              body,
              contentType
            );

          if (
            files.length === 0
          ) {
            return sendJson(
              res,
              400,
              {
                error:
                  "No file was received."
              }
            );
          }

          const added = [];

          for (
            const file of files
          ) {
            const document =
              addDocument(
                file.filename,
                file.buffer
              );

            added.push(
              publicDocument(
                document
              )
            );
          }

          const counts =
            getCounts();

          return sendJson(
            res,
            200,
            {
              ok: true,

              documents:
                added,

              documentCount:
                counts.documentCount,

              chunkCount:
                counts.chunkCount
            }
          );
        }

        /* DELETE DOCUMENT */

        if (
          req.method === "DELETE" &&
          url.pathname.startsWith(
            "/api/documents/"
          )
        ) {
          const id =
            decodeURIComponent(
              url.pathname.slice(
                "/api/documents/"
                  .length
              )
            );

          if (
            !documents.has(id)
          ) {
            return sendJson(
              res,
              404,
              {
                error:
                  "Document not found."
              }
            );
          }

          documents.delete(id);

          return sendJson(
            res,
            200,
            {
              ok: true
            }
          );
        }

        /* RESEARCH QUESTION */

        if (
          req.method === "POST" &&
          url.pathname ===
            "/api/research"
        ) {
          const body =
            await readRequestBody(
              req,
              512 * 1024
            );

          let data;

          try {
            data =
              JSON.parse(
                body.toString(
                  "utf8"
                )
              );
          } catch {
            return sendJson(
              res,
              400,
              {
                error:
                  "Invalid JSON request."
              }
            );
          }

          const question =
            cleanText(
              data.question
            );

          if (!question) {
            return sendJson(
              res,
              400,
              {
                error:
                  "Please enter a research question."
              }
            );
          }

          const result =
            await researchQuestion(
              question
            );

          return sendJson(
            res,
            200,
            result
          );
        }

        /* RESEARCH BRIEF */

        if (
          req.method === "POST" &&
          url.pathname ===
            "/api/research/brief"
        ) {
          const result =
            await researchBrief();

          return sendJson(
            res,
            200,
            result
          );
        }

        /* CONTRADICTION SCAN */

        if (
          req.method === "POST" &&
          url.pathname ===
            "/api/research/contradictions"
        ) {
          const result =
            deterministicContradictionScan();

          return sendJson(
            res,
            200,
            {
              result
            }
          );
        }

        /* FACT EXTRACTION */

        if (
          req.method === "POST" &&
          url.pathname ===
            "/api/research/facts"
        ) {
          const result =
            deterministicFactExtraction();

          return sendJson(
            res,
            200,
            {
              result
            }
          );
        }

        /* STATIC FILES */

        if (
          req.method === "GET" &&
          !url.pathname.startsWith(
            "/api/"
          )
        ) {
          let requestedPath =
            url.pathname === "/"
              ? "/index.html"
              : url.pathname;

          requestedPath =
            decodeURIComponent(
              requestedPath
            );

          const filePath =
            path.normalize(
              path.join(
                PUBLIC_DIR,
                requestedPath
              )
            );

          if (
            !filePath.startsWith(
              PUBLIC_DIR
            )
          ) {
            return sendText(
              res,
              403,
              "Forbidden"
            );
          }

          if (
            !fs.existsSync(
              filePath
            ) ||
            !fs.statSync(
              filePath
            ).isFile()
          ) {
            return sendText(
              res,
              404,
              "Not found"
            );
          }

          const extension =
            path.extname(
              filePath
            ).toLowerCase();

          const contentTypes = {
            ".html":
              "text/html; charset=utf-8",

            ".js":
              "text/javascript; charset=utf-8",

            ".css":
              "text/css; charset=utf-8",

            ".json":
              "application/json; charset=utf-8",

            ".svg":
              "image/svg+xml"
          };

          return sendText(
            res,
            200,
            fs.readFileSync(
              filePath
            ),
            contentTypes[
              extension
            ] ||
              "application/octet-stream"
          );
        }

        return sendJson(
          res,
          404,
          {
            error:
              "Not found."
          }
        );
      } catch (error) {
        console.error(error);

        return sendJson(
          res,
          500,
          {
            error:
              error?.message ||
              "Internal server error."
          }
        );
      }
    }
  );

/* =========================================================
   START SERVER
   ========================================================= */

server.listen(
  PORT,
  HOST,
  () => {
    console.log("");
    console.log(
      "=============================================="
    );
    console.log(
      " QVAC Visual Research Lab"
    );
    console.log(
      "=============================================="
    );
    console.log(
      ` Local server: http://${HOST}:${PORT}`
    );
    console.log(
      " QVAC SDK: 0.19.0"
    );
    console.log(
      " AI inference: local/on-device"
    );
    console.log(
      " Evidence guard: enabled"
    );
    console.log(
      "=============================================="
    );
    console.log("");
  }
);

/* =========================================================
   SHUTDOWN
   ========================================================= */

async function shutdown() {
  console.log(
    "\nShutting down..."
  );

  try {
    if (modelId) {
      await unloadModel({
        modelId
      });
    }
  } catch (error) {
    console.error(
      "Model unload error:",
      error.message
    );
  }

  server.close(() => {
    process.exit(0);
  });
}

process.on(
  "SIGINT",
  shutdown
);

process.on(
  "SIGTERM",
  shutdown
);