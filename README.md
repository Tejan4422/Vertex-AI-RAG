# Vertex AI RAG RFP Processor

## Overview
This cloud function automatically processes Request for Proposal (RFP) Excel sheets using Google's Vertex AI and Discovery Engine. It transforms requirements into questions, generates responses using RAG (Retrieval-Augmented Generation), and provides referenced answers from your knowledge base.

## Features
- 🤖 Automated RFP response generation
- 📝 Answers questions based on the RFP requirements
- 🔍 RAG-based answer generation using Discovery Engine
- 📚 Provides source references for each answer
- 📊 Maintains Excel formatting with proper styling
- ♻️ Automatic retry mechanism for API failures
- 🔄 Processes multiple sheets within a workbook

## Prerequisites
- Google Cloud Project with enabled APIs:
  - Vertex AI API
  - Discovery Engine API
  - Cloud Storage API
- Python 3.7+
- Required Python packages:
  ```txt
  functions-framework
  google-cloud-storage
  pandas
  vertexai
  openpyxl
  requests
  google-auth
  ```

## Setup
1. Clone this repository
2. Set up your Google Cloud project:
   ```bash
   export PROJECT_ID="your-project-id"
   gcloud config set project $PROJECT_ID
   ```
3. Enable required APIs:
   ```bash
   gcloud services enable \
       aiplatform.googleapis.com \
       discoveryengine.googleapis.com \
       storage.googleapis.com
   ```
4. Configure Discovery Engine with your knowledge base
5. Update the project configurations in the code:
   - Update `project_id` in `initialize_vertex_ai()`
   - Update Discovery Engine URL in `call_discovery_engine()`

## Usage
1. Upload your RFP Excel file to the configured Google Cloud Storage bucket
2. The cloud function automatically triggers and processes the file
3. A new file with prefix `processed_` will be created in the same bucket
4. Download the processed file containing:
   - Original requirements
   - Generated questions
   - RAG-based responses
   - Reference sources

## Excel File Format
The function supports Excel files with the following column names:
- `Bank Requirement` or `requirements` - Contains the RFP requirements
- Additional columns will be preserved, and new columns will be added:
  - `Questions` - Generated Yes/No questions
  - `Vendor Response` - RAG-generated answers
  - `References` - Source documents references

## Error Handling
- Implements exponential backoff retry mechanism
- Handles API failures gracefully
- Skips already processed files
- Validates sheet structure before processing

## Limitations
- Processing time depends on the number of requirements
- 20-second delay between RAG queries to respect API limits
- Maximum 3 reference sources per response
- Answers are limited to 100-200 words

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License
[Add your chosen license here]

## Authors
[Add your name/organization here]

## Acknowledgments
- Google Vertex AI
- Google Discovery Engine
- Cloud Functions Framework
