/**
 * Tests for OllamaClient
 */

import { OllamaClient, OllamaClientConfig } from '../../src/api/ollama-client';
import { BaseModelRequest, ExtendedModelRequest } from '../../src/api/interfaces/model-api';
import { ModelProvider } from '../../src/api/types';
import { TextEncoder, TextDecoder } from 'util';

// Make TextEncoder and TextDecoder available globally for tests
global.TextEncoder = TextEncoder as any;
global.TextDecoder = TextDecoder as any;

// Mock fetch globally
global.fetch = jest.fn();

describe('OllamaClient', () => {
	let client: OllamaClient;
	let config: OllamaClientConfig;

	beforeEach(() => {
		config = {
			endpoint: 'http://localhost:11434',
			model: 'llama3.1',
			temperature: 0.7,
			topP: 1.0,
		};
		client = new OllamaClient(config);
		jest.clearAllMocks();
	});

	describe('getCapabilities', () => {
		it('should return correct capabilities for regular model', () => {
			const caps = client.getCapabilities('llama3.1');

			expect(caps.supportsVision).toBe(false);
			expect(caps.supportsGrounding).toBe(false);
			expect(caps.supportsToolCalling).toBe(false);
			expect(caps.supportsCloudRag).toBe(false);
			expect(caps.supportsLocalRag).toBe(true);
		});

		it('should detect llava vision models', () => {
			const caps = client.getCapabilities('llava-7b');

			expect(caps.supportsVision).toBe(true);
			expect(caps.supportsLocalRag).toBe(true);
		});
	});

	describe('generateModelResponse', () => {
		it('should handle BaseModelRequest', async () => {
			const mockResponse = {
				model: 'llama3.1',
				message: {
					role: 'assistant',
					content: 'Test response',
				},
				done: true,
			};

			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse,
			});

			const request: BaseModelRequest = {
				prompt: 'Test prompt',
			};

			const response = await client.generateModelResponse(request);

			expect(response.markdown).toBe('Test response');
			expect(response.rendered).toBe('');
			expect(global.fetch).toHaveBeenCalledWith(
				'http://localhost:11434/api/chat',
				expect.objectContaining({
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
				})
			);
		});

		it('should handle ExtendedModelRequest with conversation history', async () => {
			const mockResponse = {
				model: 'llama3.1',
				message: {
					role: 'assistant',
					content: 'Response with history',
				},
				done: true,
			};

			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse,
			});

			const request: ExtendedModelRequest = {
				prompt: 'System instruction',
				conversationHistory: [
					{
						role: 'user',
						parts: [{ text: 'Hello' }],
					},
					{
						role: 'model',
						parts: [{ text: 'Hi there!' }],
					},
				],
				userMessage: 'How are you?',
			};

			const response = await client.generateModelResponse(request);

			expect(response.markdown).toBe('Response with history');
			expect(global.fetch).toHaveBeenCalled();

			// Check the request body
			const callArgs = (global.fetch as jest.Mock).mock.calls[0];
			const requestBody = JSON.parse(callArgs[1].body);

			expect(requestBody.messages).toHaveLength(4); // system + 2 history + user
			expect(requestBody.messages[0].role).toBe('system');
			expect(requestBody.messages[1].role).toBe('user');
			expect(requestBody.messages[1].content).toBe('Hello');
			expect(requestBody.messages[2].role).toBe('assistant');
			expect(requestBody.messages[2].content).toBe('Hi there!');
			expect(requestBody.messages[3].role).toBe('user');
			expect(requestBody.messages[3].content).toBe('How are you?');
		});

		it('should handle API errors', async () => {
			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: false,
				status: 500,
				statusText: 'Internal Server Error',
				text: async () => 'Server error',
			});

			const request: BaseModelRequest = {
				prompt: 'Test prompt',
			};

			await expect(client.generateModelResponse(request)).rejects.toThrow('Ollama API error');
		});
	});

	describe('generateEmbedding', () => {
		it('should generate embeddings', async () => {
			const mockEmbedding = [0.1, 0.2, 0.3, 0.4, 0.5];
			const mockResponse = {
				embedding: mockEmbedding,
			};

			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse,
			});

			const embedding = await client.generateEmbedding('Test text', 'nomic-embed-text');

			expect(embedding).toEqual(mockEmbedding);
			expect(global.fetch).toHaveBeenCalledWith(
				'http://localhost:11434/api/embeddings',
				expect.objectContaining({
					method: 'POST',
				})
			);

			const callArgs = (global.fetch as jest.Mock).mock.calls[0];
			const requestBody = JSON.parse(callArgs[1].body);
			expect(requestBody.model).toBe('nomic-embed-text');
			expect(requestBody.prompt).toBe('Test text');
		});

		it('should use default model if not specified', async () => {
			const mockResponse = {
				embedding: [0.1, 0.2],
			};

			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse,
			});

			await client.generateEmbedding('Test text');

			const callArgs = (global.fetch as jest.Mock).mock.calls[0];
			const requestBody = JSON.parse(callArgs[1].body);
			expect(requestBody.model).toBe('llama3.1'); // Falls back to config.model
		});
	});

	describe('generateStreamingResponse', () => {
		it('should handle streaming responses', async () => {
			const chunks = [
				{ message: { content: 'Hello' }, done: false },
				{ message: { content: ' world' }, done: false },
				{ message: { content: '!' }, done: true },
			];

			const mockReader = {
				read: jest
					.fn()
					.mockResolvedValueOnce({
						done: false,
						value: new TextEncoder().encode(JSON.stringify(chunks[0]) + '\n'),
					})
					.mockResolvedValueOnce({
						done: false,
						value: new TextEncoder().encode(JSON.stringify(chunks[1]) + '\n'),
					})
					.mockResolvedValueOnce({
						done: false,
						value: new TextEncoder().encode(JSON.stringify(chunks[2]) + '\n'),
					})
					.mockResolvedValueOnce({
						done: true,
						value: undefined,
					}),
				cancel: jest.fn(),
			};

			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: true,
				body: {
					getReader: () => mockReader,
				},
			});

			const request: BaseModelRequest = {
				prompt: 'Test prompt',
			};

			const receivedChunks: string[] = [];
			const streamResponse = client.generateStreamingResponse(request, (chunk) => {
				receivedChunks.push(chunk.text);
			});

			const response = await streamResponse.complete;

			expect(receivedChunks).toEqual(['Hello', ' world', '!']);
			expect(response.markdown).toBe('Hello world!');
			expect(response.rendered).toBe('');
		});

		it('should handle cancellation', async () => {
			let readCallCount = 0;
			const mockCancelReader = {
				read: jest.fn().mockImplementation(() => {
					readCallCount++;
					return new Promise((resolve) => {
						setTimeout(() => {
							if (readCallCount === 1) {
								resolve({
									done: false,
									value: new TextEncoder().encode('{"message":{"content":"test"},"done":false}\n'),
								});
							} else {
								resolve({
									done: true,
									value: undefined,
								});
							}
						}, 50);
					});
				}),
				cancel: jest.fn().mockResolvedValue(undefined),
			};

			(global.fetch as jest.Mock).mockResolvedValueOnce({
				ok: true,
				body: {
					getReader: () => mockCancelReader,
				},
			});

			const request: BaseModelRequest = {
				prompt: 'Test prompt',
			};

			const streamResponse = client.generateStreamingResponse(request, () => {});

			// Cancel after a short delay
			setTimeout(() => streamResponse.cancel(), 25);

			const response = await streamResponse.complete;

			// Cancel should have been called
			expect(mockCancelReader.cancel).toHaveBeenCalled();
		});
	});
});
