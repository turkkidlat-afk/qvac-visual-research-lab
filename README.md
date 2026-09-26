# QVAC Visual Research Lab

QVAC Visual Research Lab is a local AI research workstation for investigating information across multiple documents.

It lets users upload local documents, ask research questions, inspect supporting evidence, generate research briefs, extract facts, and scan for possible contradictions.

AI inference for research questions runs locally on the user's computer using Tether QVAC.

## Features

- Upload multiple TXT, Markdown, JSON, and CSV documents
- Search across local document evidence
- Ask research questions using local AI
- Display supporting evidence sources
- Generate research briefs
- Extract structured facts
- Scan documents for possible contradictions
- Keep document processing and AI inference local
- No cloud AI API key required

## QVAC SDK

- QVAC SDK: `@qvac/sdk` version `0.19.0`
- Model: `LLAMA_3_2_1B_INST_Q4_0`

The application calls:

- `loadModel()`
- `completion()`
- `unloadModel()`

## Requirements

- Node.js 18 or newer
- A computer capable of running the QVAC local model
- Internet access for the initial QVAC model download

## Installation

Clone the repository:

`git clone https://github.com/turkkidlat-afk/qvac-visual-research-lab.git`

Enter the project directory:

`cd qvac-visual-research-lab`

Install dependencies:

`npm install`

## Run

Start the application:

`npm start`

Then open:

`http://127.0.0.1:4028`

## How It Works

1. Add local documents to the Research Library.
2. The application reads and chunks the document content locally.
3. Enter a research question.
4. Relevant evidence is selected from the local document collection.
5. QVAC loads the local language model when needed.
6. QVAC generates the research answer on-device.
7. The answer and supporting evidence are displayed in the interface.

The Research Brief, Fact Extraction, and Contradiction Scan tools analyze the local document collection without sending documents to a cloud AI service.

## Supported Documents

- `.txt`
- `.md`
- `.json`
- `.csv`

Individual file uploads are limited to 2 MB.

## Example Research Questions

- What are the main environmental concerns reported by residents, and what actions are proposed to address them?
- What evidence shows that plastic waste is a major concern in the community?
- How do the river cleanup report, community survey, and cleanup proposal relate to each other?
- What specific actions are proposed to reduce plastic waste and litter around Riverside Creek?
- What numbers and dates are mentioned across the three documents?

## Research Tools

### Research Question

Uses QVAC to answer questions based on evidence retrieved from the local document collection.

### Research Brief

Creates a concise summary of important findings contained in the local documents.

### Fact Extraction

Extracts structured facts such as names, dates, quantities, locations, and other details found in the documents.

### Contradiction Scan

Checks the local evidence for statements that may conflict with each other and identifies the relevant source material.

## Local AI

QVAC Visual Research Lab uses Tether QVAC to run its language model directly on the user's computer.

The main QVAC workflow uses `loadModel()`, `completion()`, and `unloadModel()`.

## Privacy

The application is designed around local processing.

Documents are processed by the local application while the server is running. Research-question inference is performed locally using QVAC.

No external AI API key is required.

## Project Structure

- `public/index.html` — web interface
- `server.js` — local server and QVAC integration
- `package.json` — project configuration and dependencies
- `package-lock.json` — dependency lockfile
- `.gitignore` — ignored files
- `README.md` — project documentation

## License

MIT License
