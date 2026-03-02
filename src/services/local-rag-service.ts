/**
 * Local RAG Service using Ollama embeddings
 *
 * Implements semantic search and indexing using local vector embeddings
 * from Ollama, stored in local JSON files for privacy and offline access.
 */

import { TFile, normalizePath, Notice } from 'obsidian';
import type ObsidianGemini from '../main';
import { OllamaClient, OllamaClientConfig } from '../api/ollama-client';

// Import TextEncoder for Node.js environment
let TextEncoder: any;
if (typeof window === 'undefined') {
	TextEncoder = require('util').TextEncoder;
}

/**
 * Represents an indexed document chunk
 */
export interface DocumentChunk {
	path: string;
	content: string;
	contentHash: string; // SHA-256 for change detection
	embedding: number[];
	timestamp: number;
}

/**
 * Local RAG cache structure
 */
export interface LocalRagCache {
	version: string;
	lastSync: number;
	embeddingModel: string;
	documents: Record<string, DocumentChunk[]>; // path -> chunks
}

/**
 * Search result from local RAG
 */
export interface LocalRagSearchResult {
	path: string;
	content: string;
	relevance: number; // 0-1 cosine similarity score
}

/**
 * Index progress information
 */
export interface LocalRagProgress {
	phase: 'scanning' | 'embedding' | 'complete' | 'error';
	current: number;
	total: number;
	currentFile?: string;
	message?: string;
}

const CACHE_VERSION = '1.0';
const MIN_CHUNK_SIZE = 100;
const MAX_CHUNK_SIZE = 2000;

/**
 * LocalRagService - Vector-based semantic search using Ollama embeddings
 *
 * Mirrors RagIndexingService interface but uses local embeddings instead of
 * Google's File Search API. Enables completely private, offline RAG.
 */
export class LocalRagService {
	private plugin: ObsidianGemini;
	private ollamaClient: OllamaClient | null = null;
	private cache: LocalRagCache | null = null;
	private isIndexing: boolean = false;
	private cancelRequested: boolean = false;

	constructor(plugin: ObsidianGemini) {
		this.plugin = plugin;
	}

	/**
	 * Get cache file path
	 */
	private get cachePath(): string {
		return normalizePath(`${this.plugin.settings.historyFolder}/local-rag-cache.json`);
	}

	/**
	 * Initialize the service
	 */
	async initialize(): Promise<void> {
		const ollamaConfig: OllamaClientConfig = {
			endpoint: this.plugin.settings.ollama.endpoint,
			model: this.plugin.settings.ollama.models.embedding,
		};

		this.ollamaClient = new OllamaClient(ollamaConfig, undefined, this.plugin);

		// Load existing cache
		await this.loadCache();

		this.plugin.logger.log('LocalRagService: Initialized successfully');
	}

	/**
	 * Destroy the service
	 */
	async destroy(): Promise<void> {
		this.cancelRequested = true;
		if (this.isIndexing) {
			this.plugin.logger.log('LocalRagService: Cancelling indexing during destroy');
		}
	}

	/**
	 * Index all markdown files in vault
	 */
	async indexVault(): Promise<{ indexed: number; skipped: number; failed: number }> {
		if (!this.ollamaClient) {
			throw new Error('LocalRagService not initialized');
		}

		if (!this.plugin.settings.ollama.models.embedding) {
			throw new Error('No embedding model configured');
		}

		this.isIndexing = true;
		this.cancelRequested = false;

		let indexed = 0;
		let skipped = 0;
		let failed = 0;

		try {
			// Get all markdown files
			const mdFiles = this.plugin.app.vault.getMarkdownFiles();
			const filesToIndex = mdFiles.filter((f) => !this.isSystemFile(f.path));

			this.plugin.logger.log(`LocalRagService: Indexing ${filesToIndex.length} files`);

			for (let i = 0; i < filesToIndex.length; i++) {
				if (this.cancelRequested) break;

				const file = filesToIndex[i];

				try {
					const content = await this.plugin.app.vault.read(file);
					const contentHash = await this.hashContent(content);

					// Check if file needs re-indexing
					const existingChunks = this.cache?.documents[file.path] || [];
					if (existingChunks.length > 0 && existingChunks[0]?.contentHash === contentHash) {
						skipped++;
						continue;
					}

					// Split content into chunks
					const chunks = this.chunkContent(content);

					// Generate embeddings for each chunk
					const embeddings: DocumentChunk[] = [];
					for (const chunk of chunks) {
						if (this.cancelRequested) break;

						const embedding = await this.ollamaClient.generateEmbedding(
							chunk,
							this.plugin.settings.ollama.models.embedding
						);

						embeddings.push({
							path: file.path,
							content: chunk,
							contentHash,
							embedding,
							timestamp: Date.now(),
						});
					}

					// Store in cache
					if (!this.cache) {
						this.cache = {
							version: CACHE_VERSION,
							lastSync: Date.now(),
							embeddingModel: this.plugin.settings.ollama.models.embedding,
							documents: {},
						};
					}

					this.cache.documents[file.path] = embeddings;
					indexed++;
				} catch (error) {
					this.plugin.logger.error(`LocalRagService: Failed to index ${file.path}:`, error);
					failed++;
				}
			}

			// Save cache
			await this.saveCache();

			this.plugin.logger.log(
				`LocalRagService: Indexing complete - ${indexed} indexed, ${skipped} skipped, ${failed} failed`
			);

			return { indexed, skipped, failed };
		} finally {
			this.isIndexing = false;
		}
	}

	/**
	 * Search indexed documents using semantic similarity
	 */
	async search(query: string, maxResults: number = 5): Promise<LocalRagSearchResult[]> {
		if (!this.ollamaClient) {
			throw new Error('LocalRagService not initialized');
		}

		if (!this.cache || Object.keys(this.cache.documents).length === 0) {
			return [];
		}

		// Generate embedding for query
		const queryEmbedding = await this.ollamaClient.generateEmbedding(
			query,
			this.plugin.settings.ollama.models.embedding
		);

		// Score all chunks
		const scored: Array<{ chunk: DocumentChunk; score: number }> = [];

		for (const chunks of Object.values(this.cache.documents)) {
			for (const chunk of chunks) {
				const score = this.cosineSimilarity(queryEmbedding, chunk.embedding);
				scored.push({ chunk, score });
			}
		}

		// Sort by relevance and return top results
		return scored
			.sort((a, b) => b.score - a.score)
			.slice(0, maxResults)
			.map(({ chunk, score }) => ({
				path: chunk.path,
				content: chunk.content,
				relevance: score,
			}));
	}

	/**
	 * Delete the index
	 */
	async deleteIndex(): Promise<void> {
		try {
			const file = this.plugin.app.vault.getAbstractFileByPath(this.cachePath);
			if (file && file instanceof TFile) {
				await this.plugin.app.vault.delete(file);
			}
			this.cache = null;
			this.plugin.logger.log('LocalRagService: Index deleted');
		} catch (error) {
			this.plugin.logger.warn('LocalRagService: Failed to delete index:', error);
		}
	}

	/**
	 * Get index statistics
	 */
	getStats(): { fileCount: number; chunkCount: number; model: string } {
		let fileCount = 0;
		let chunkCount = 0;

		if (this.cache) {
			fileCount = Object.keys(this.cache.documents).length;
			for (const chunks of Object.values(this.cache.documents)) {
				chunkCount += chunks.length;
			}
		}

		return {
			fileCount,
			chunkCount,
			model: this.cache?.embeddingModel || '',
		};
	}

	/**
	 * Check if file is a system file
	 */
	private isSystemFile(path: string): boolean {
		return (
			path.startsWith('.obsidian/') ||
			path.startsWith(this.plugin.settings.historyFolder + '/') ||
			path.startsWith(this.plugin.settings.ragIndexing.excludeFolders.join('/'))
		);
	}

	/**
	 * Split content into overlapping chunks
	 */
	private chunkContent(content: string): string[] {
		const chunks: string[] = [];
		const paragraphs = content.split('\n\n').filter((p) => p.trim().length > 0);

		let currentChunk = '';

		for (const para of paragraphs) {
			const potential = currentChunk ? currentChunk + '\n\n' + para : para;

			if (potential.length <= MAX_CHUNK_SIZE) {
				currentChunk = potential;
			} else {
				if (currentChunk.length >= MIN_CHUNK_SIZE) {
					chunks.push(currentChunk);
				}
				currentChunk = para;
			}
		}

		if (currentChunk.length >= MIN_CHUNK_SIZE) {
			chunks.push(currentChunk);
		}

		return chunks;
	}

	/**
	 * Cosine similarity between two vectors
	 */
	private cosineSimilarity(a: number[], b: number[]): number {
		if (a.length !== b.length) return 0;

		let dotProduct = 0;
		let magnitudeA = 0;
		let magnitudeB = 0;

		for (let i = 0; i < a.length; i++) {
			dotProduct += a[i] * b[i];
			magnitudeA += a[i] * a[i];
			magnitudeB += b[i] * b[i];
		}

		magnitudeA = Math.sqrt(magnitudeA);
		magnitudeB = Math.sqrt(magnitudeB);

		if (magnitudeA === 0 || magnitudeB === 0) return 0;

		return dotProduct / (magnitudeA * magnitudeB);
	}

	/**
	 * Hash content for change detection
	 */
	private async hashContent(content: string): Promise<string> {
		// Use Node.js crypto for Node environment, browser crypto for browser
		try {
			// Try Node.js crypto first
			const crypto = require('crypto');
			return crypto.createHash('sha256').update(content).digest('hex');
		} catch {
			// Fallback to browser crypto
			const encoder = new (globalThis.TextEncoder as any)();
			const data = encoder.encode(content);
			const hashBuffer = await crypto.subtle.digest('SHA-256', data);
			const hashArray = Array.from(new Uint8Array(hashBuffer));
			return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
		}
	}

	/**
	 * Load cache from disk
	 */
	private async loadCache(): Promise<void> {
		try {
			const file = this.plugin.app.vault.getAbstractFileByPath(this.cachePath);
			if (file && file instanceof TFile) {
				const content = await this.plugin.app.vault.read(file);
				this.cache = JSON.parse(content);
				this.plugin.logger.log('LocalRagService: Cache loaded');
			}
		} catch (error) {
			this.plugin.logger.debug('LocalRagService: No existing cache found:', error);
			this.cache = null;
		}
	}

	/**
	 * Save cache to disk
	 */
	private async saveCache(): Promise<void> {
		if (!this.cache) return;

		try {
			const content = JSON.stringify(this.cache, null, 2);
			const file = this.plugin.app.vault.getAbstractFileByPath(this.cachePath);

			if (file && file instanceof TFile) {
				await this.plugin.app.vault.modify(file, content);
			} else {
				await this.plugin.app.vault.create(this.cachePath, content);
			}
		} catch (error) {
			this.plugin.logger.error('LocalRagService: Failed to save cache:', error);
		}
	}
}
