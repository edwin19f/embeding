/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';

// Get API Key from environment variables
const API_KEY = process.env.API_KEY;

// DOM element references
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const embedButton = document.getElementById('embed-button') as HTMLButtonElement;
const outputContainer = document.getElementById('output-container') as HTMLDivElement;
const outputElement = document.getElementById('output') as HTMLPreElement;
const saveButton = document.getElementById('save-button') as HTMLButtonElement;
const downloadButton = document.getElementById('download-button') as HTMLButtonElement;
const comparisonContainer = document.getElementById('comparison-container') as HTMLDivElement;
const storedFileNameElement = document.getElementById('stored-file-name') as HTMLSpanElement;
const compareButton = document.getElementById('compare-button') as HTMLButtonElement;
const similarityScoreElement = document.getElementById('similarity-score') as HTMLParagraphElement;

// Ensure all required elements are found
if (!fileInput || !embedButton || !outputContainer || !outputElement || !saveButton || !downloadButton || !comparisonContainer || !storedFileNameElement || !compareButton || !similarityScoreElement) {
  throw new Error('Required DOM elements are missing.');
}

// --- State Management ---
type Embedding = {
  fileName: string;
  values: number[];
};

let ai: GoogleGenAI;
let currentEmbedding: Embedding | null = null;
let storedEmbedding: Embedding | null = null;
const STORED_EMBEDDING_KEY = 'gemini-embedding-stored';

// --- Initialization ---
try {
  ai = new GoogleGenAI({ apiKey: API_KEY });
} catch (e) {
  console.error(e);
  outputElement.textContent = 'ERROR: Failed to initialize Gemini API. Make sure the API_KEY environment variable is set correctly.';
  outputContainer.style.display = 'block';
  embedButton.disabled = true;
  fileInput.disabled = true;
}

// --- Utility Functions ---

/** Reads the content of a file as text. */
function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

/** Calculates cosine similarity between two vectors. */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error('Vectors must have the same dimension.');
  }
  let dotProduct = 0;
  let magA = 0;
  let magB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    magA += vecA[i] * vecA[i];
    magB += vecB[i] * vecB[i];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  if (magA === 0 || magB === 0) {
    return 0; // Or handle as an error, depending on context
  }
  return dotProduct / (magA * magB);
}

// --- UI Update Functions ---

/** Updates the UI for the stored embedding section. */
function updateStoredEmbeddingUI() {
  if (storedEmbedding) {
    storedFileNameElement.textContent = storedEmbedding.fileName;
    comparisonContainer.style.display = 'block';
  } else {
    comparisonContainer.style.display = 'none';
  }
  updateCompareButtonState();
}

/** Updates the enabled/disabled state of the compare button. */
function updateCompareButtonState() {
    compareButton.disabled = !(currentEmbedding && storedEmbedding);
}

/** Resets the current output and comparison UI. */
function resetCurrentOutput() {
    currentEmbedding = null;
    outputContainer.style.display = 'none';
    outputElement.textContent = '';
    saveButton.style.display = 'none';
    downloadButton.style.display = 'none';
    similarityScoreElement.textContent = '';
    updateCompareButtonState();
}

// --- Local Storage Functions ---

/** Saves the current embedding to local storage. */
function saveEmbeddingForComparison() {
    if (!currentEmbedding) return;
    storedEmbedding = currentEmbedding;
    localStorage.setItem(STORED_EMBEDDING_KEY, JSON.stringify(storedEmbedding));
    updateStoredEmbeddingUI();
    alert(`Embedding for "${currentEmbedding.fileName}" is now stored for comparison.`);
}

/** Loads stored embedding from local storage on page load. */
function loadStoredEmbedding() {
    const stored = localStorage.getItem(STORED_EMBEDDING_KEY);
    if (stored) {
        try {
            storedEmbedding = JSON.parse(stored);
            updateStoredEmbeddingUI();
        } catch (e) {
            console.error('Failed to parse stored embedding:', e);
            localStorage.removeItem(STORED_EMBEDDING_KEY);
        }
    }
}


// --- Event Handlers ---

/** Main function to handle the embedding process. */
async function handleEmbed() {
  if (!ai) {
    outputElement.textContent = 'Gemini API is not initialized.';
    outputContainer.style.display = 'block';
    return;
  }

  const file = fileInput.files?.[0];
  if (!file) {
    alert('Please select a file first.');
    return;
  }

  resetCurrentOutput();
  embedButton.disabled = true;
  embedButton.textContent = 'Embedding...';
  outputContainer.style.display = 'block';
  outputElement.textContent = 'Generating embedding...';

  try {
    const fileContent = await readFileAsText(file);

    const response = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: fileContent,
    });
    
    const embeddingValues = response.embeddings[0].values;
    currentEmbedding = { fileName: file.name, values: embeddingValues };

    outputElement.textContent = `Embedding for "${file.name}" successful!\n\n`;
    outputElement.textContent += JSON.stringify(embeddingValues, null, 2);
    saveButton.style.display = 'block';
    downloadButton.style.display = 'block';
    updateCompareButtonState();

  } catch (error) {
    console.error('Embedding failed:', error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    outputElement.textContent = `An error occurred: ${errorMessage}`;
  } finally {
    embedButton.disabled = false;
    embedButton.textContent = 'Embed Content';
  }
}

/** Handles the comparison logic. */
function handleCompare() {
    if (!currentEmbedding || !storedEmbedding) {
        alert('Both a current and a stored embedding are required to compare.');
        return;
    }
    try {
        const similarity = cosineSimilarity(currentEmbedding.values, storedEmbedding.values);
        similarityScoreElement.textContent = `Similarity Score: ${similarity.toFixed(4)}`;
    } catch (error) {
        console.error('Comparison failed:', error);
        similarityScoreElement.textContent = error instanceof Error ? `Error: ${error.message}` : 'An unknown error occurred.';
    }
}

/** Handles downloading the current embedding as a JSON file. */
function handleDownload() {
    if (!currentEmbedding) {
        alert('No current embedding to download.');
        return;
    }

    const dataStr = JSON.stringify(currentEmbedding, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `embedding-${currentEmbedding.fileName}.json`;
    document.body.appendChild(link);
    link.click();
    
    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// --- Attach Event Listeners ---
document.addEventListener('DOMContentLoaded', loadStoredEmbedding);
embedButton.addEventListener('click', handleEmbed);
saveButton.addEventListener('click', saveEmbeddingForComparison);
compareButton.addEventListener('click', handleCompare);
downloadButton.addEventListener('click', handleDownload);
fileInput.addEventListener('change', () => {
    if (fileInput.files?.length) {
        resetCurrentOutput();
    }
});