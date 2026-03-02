/**
 * Tests for LocalRagService
 */

import { TextEncoder } from 'util';
import { LocalRagService, LocalRagCache, DocumentChunk } from '../../src/services/local-rag-service';

// Mock crypto for Node.js
if (typeof global.crypto === 'undefined') {
	global.crypto = require('crypto').webcrypto as any;
}

describe('LocalRagService', () => {
	let service: LocalRagService;

	// Mock plugin
	const mockPlugin = {
		settings: {
			historyFolder: 'gemini-scribe',
			ollama: {
				endpoint: 'http://localhost:11434',
				models: {
					embedding: 'nomic-embed-text',
				},
			},
			ragIndexing: {
				excludeFolders: ['.obsidian'],
			},
		},
		app: {
			vault: {
				getAbstractFileByPath: jest.fn(),
				getMarkdownFiles: jest.fn(() => []),
				read: jest.fn(),
				create: jest.fn(),
				modify: jest.fn(),
				delete: jest.fn(),
			},
		},
		logger: {
			log: jest.fn(),
			debug: jest.fn(),
			warn: jest.fn(),
			error: jest.fn(),
		},
	} as any;

	beforeEach(() => {
		jest.clearAllMocks();
		service = new LocalRagService(mockPlugin);
	});

	describe('chunkContent', () => {
		it('should split content into chunks by paragraph', () => {
			const content = `First paragraph with some text that is long enough to meet the minimum chunk size requirement for indexing purposes.

Second paragraph with more text that has enough content to be considered a full chunk for the search index.

Third paragraph here that also contains sufficient content to meet minimum requirements and be indexed separately.`;

			const chunks = (service as any).chunkContent(content);

			expect(chunks.length).toBeGreaterThan(0);
			expect(chunks[0]).toContain('First');
		});

		it('should skip chunks smaller than MIN_CHUNK_SIZE', () => {
			const content = `Short.

This is a much longer paragraph with enough content to meet the minimum chunk size requirement for proper indexing.`;

			const chunks = (service as any).chunkContent(content);

			expect(chunks.length).toBeGreaterThan(0);
			chunks.forEach((chunk: string) => {
				expect(chunk.length).toBeGreaterThanOrEqual(100);
			});
		});

		it('should limit chunks to MAX_CHUNK_SIZE', () => {
			const longPara = 'Word '.repeat(300); // Create a very long paragraph
			const content = longPara + '\n\n' + longPara;

			const chunks = (service as any).chunkContent(content);

			expect(chunks.length).toBeGreaterThan(0);
			chunks.forEach((chunk: string) => {
				expect(chunk.length).toBeLessThanOrEqual(2000);
			});
		});

		it('should handle empty content', () => {
			const chunks = (service as any).chunkContent('');
			expect(chunks.length).toBe(0);
		});

		it('should handle content with only whitespace', () => {
			const chunks = (service as any).chunkContent('   \n\n   \n\n  ');
			expect(chunks.length).toBe(0);
		});
	});

	describe('cosineSimilarity', () => {
		it('should compute identical vectors as 1.0', () => {
			const vec = [1, 0, 1, 1];
			const similarity = (service as any).cosineSimilarity(vec, vec);

			expect(similarity).toBeCloseTo(1.0);
		});

		it('should compute orthogonal vectors as 0.0', () => {
			const vecA = [1, 0];
			const vecB = [0, 1];
			const similarity = (service as any).cosineSimilarity(vecA, vecB);

			expect(similarity).toBe(0);
		});

		it('should return 0 for zero vectors', () => {
			const vecA = [0, 0, 0];
			const vecB = [1, 2, 3];
			const similarity = (service as any).cosineSimilarity(vecA, vecB);

			expect(similarity).toBe(0);
		});

		it('should compute partial similarity correctly', () => {
			const vecA = [1, 1];
			const vecB = [1, 0];
			const similarity = (service as any).cosineSimilarity(vecA, vecB);

			// (1*1 + 1*0) / (sqrt(2) * 1) = 1/sqrt(2) ≈ 0.707
			expect(similarity).toBeCloseTo(0.707, 2);
		});
	});

	describe('isSystemFile', () => {
		it('should identify .obsidian files as system files', () => {
			const isSystem = (service as any).isSystemFile('.obsidian/config.json');
			expect(isSystem).toBe(true);
		});

		it('should identify history folder files as system files', () => {
			const isSystem = (service as any).isSystemFile('gemini-scribe/history.md');
			expect(isSystem).toBe(true);
		});

		it('should not identify regular notes as system files', () => {
			const isSystem = (service as any).isSystemFile('notes/my-note.md');
			expect(isSystem).toBe(false);
		});
	});

	describe('getStats', () => {
		it('should return zero stats when no cache', () => {
			const stats = service.getStats();

			expect(stats.fileCount).toBe(0);
			expect(stats.chunkCount).toBe(0);
			expect(stats.model).toBe('');
		});

		it('should return accurate stats with cache', () => {
			const cache: LocalRagCache = {
				version: '1.0',
				lastSync: Date.now(),
				embeddingModel: 'nomic-embed-text',
				documents: {
					'file1.md': [
						{
							path: 'file1.md',
							content: 'chunk1',
							contentHash: 'hash1',
							embedding: [0.1, 0.2],
							timestamp: Date.now(),
						},
						{
							path: 'file1.md',
							content: 'chunk2',
							contentHash: 'hash1',
							embedding: [0.3, 0.4],
							timestamp: Date.now(),
						},
					],
					'file2.md': [
						{
							path: 'file2.md',
							content: 'chunk3',
							contentHash: 'hash2',
							embedding: [0.5, 0.6],
							timestamp: Date.now(),
						},
					],
				},
			};

			(service as any).cache = cache;

			const stats = service.getStats();

			expect(stats.fileCount).toBe(2);
			expect(stats.chunkCount).toBe(3);
			expect(stats.model).toBe('nomic-embed-text');
		});
	});

	describe('hashContent', () => {
		it('should generate consistent hashes', async () => {
			const content = 'Test content';
			const hash1 = await (service as any).hashContent(content);
			const hash2 = await (service as any).hashContent(content);

			expect(hash1).toBe(hash2);
			expect(hash1.length).toBe(64); // SHA-256 hex string length
		});

		it('should generate different hashes for different content', async () => {
			const hash1 = await (service as any).hashContent('Content A');
			const hash2 = await (service as any).hashContent('Content B');

			expect(hash1).not.toBe(hash2);
		});
	});
});
