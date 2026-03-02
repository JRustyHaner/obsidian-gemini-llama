/**
 * Tests for GeminiClientFactory provider routing
 */

import { GeminiClientFactory, ModelUseCase } from '../../src/api/simple-factory';
import { ModelProvider, RagProvider } from '../../src/api/types';

describe('GeminiClientFactory', () => {
	let mockPlugin: any;

	beforeEach(() => {
		mockPlugin = {
			apiKey: 'test-api-key',
			settings: {
				chatModelName: 'gemini-2.0-flash',
				summaryModelName: 'gemini-2.0-flash',
				completionsModelName: 'gemini-2.0-flash',
				imageModelName: 'gemini-2.0-flash',
				temperature: 0.7,
				topP: 0.95,
				streamingEnabled: true,
				maxRetries: 3,
				initialBackoffDelay: 1000,
				// Provider selection defaults
				chatProvider: ModelProvider.GEMINI,
				summaryProvider: ModelProvider.GEMINI,
				completionsProvider: ModelProvider.GEMINI,
				rewriteProvider: ModelProvider.GEMINI,
				// Ollama settings
				ollama: {
					enabled: false,
					endpoint: '',
					models: {
						chat: '',
						summary: '',
						completions: '',
						rewrite: '',
						embedding: '',
					},
				},
				// RAG settings
				ragIndexing: {
					enabled: true,
					fileSearchStoreName: null,
					provider: RagProvider.GEMINI,
					excludeFolders: [],
					autoSync: true,
					includeAttachments: false,
				},
			},
			logger: {
				log: jest.fn(),
				debug: jest.fn(),
				warn: jest.fn(),
				error: jest.fn(),
			},
			getModelManager: jest.fn(),
			ragIndexing: {
				indexVault: jest.fn(),
			},
		};
	});

	describe('Provider Resolution', () => {
		it('should return GEMINI provider by default for all use cases', () => {
			// Test each use case
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.CHAT)).toBe(ModelProvider.GEMINI);
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.SUMMARY)).toBe(ModelProvider.GEMINI);
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.COMPLETIONS)).toBe(ModelProvider.GEMINI);
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.REWRITE)).toBe(ModelProvider.GEMINI);
		});

		it('should always return GEMINI for SEARCH use case (requires tool calling)', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.SEARCH)).toBe(ModelProvider.GEMINI);
		});

		it('should return OLLAMA when configured and enabled for text use cases', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.CHAT)).toBe(ModelProvider.OLLAMA);
		});

		it('should fallback to GEMINI when OLLAMA selected but not enabled', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = false;

			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.CHAT)).toBe(ModelProvider.GEMINI);
		});

		it('should fallback to GEMINI when OLLAMA selected but endpoint not configured', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = '';

			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.CHAT)).toBe(ModelProvider.GEMINI);
		});

		it('should respect per-feature provider selection', () => {
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			// Set different providers per feature
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.summaryProvider = ModelProvider.GEMINI;
			mockPlugin.settings.completionsProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.rewriteProvider = ModelProvider.GEMINI;

			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.CHAT)).toBe(ModelProvider.OLLAMA);
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.SUMMARY)).toBe(ModelProvider.GEMINI);
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.COMPLETIONS)).toBe(ModelProvider.OLLAMA);
			expect(GeminiClientFactory['resolveProvider'](mockPlugin, ModelUseCase.REWRITE)).toBe(ModelProvider.GEMINI);
		});
	});

	describe('Client Creation', () => {
		it('should create GeminiClient when GEMINI provider is selected', () => {
			mockPlugin.settings.chatProvider = ModelProvider.GEMINI;

			const client = GeminiClientFactory.createFromPlugin(mockPlugin, ModelUseCase.CHAT);

			// Verify client is wrapped with RetryDecorator (should have the wrapper interface)
			expect(client).toBeDefined();
			expect(typeof client.generateModelResponse).toBe('function');
		});

		it('should create OllamaClient when OLLAMA provider is selected and configured', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';
			mockPlugin.settings.ollama.models.chat = 'llama2';

			const client = GeminiClientFactory.createFromPlugin(mockPlugin, ModelUseCase.CHAT);

			// Verify client is created and wrapped
			expect(client).toBeDefined();
			expect(typeof client.generateModelResponse).toBe('function');
		});

		it('should fallback to GEMINI when creating OLLAMA client but OLLAMA not properly configured', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = false; // Not enabled

			const client = GeminiClientFactory.createFromPlugin(mockPlugin, ModelUseCase.CHAT);

			// Client should still be created (as Gemini fallback)
			expect(client).toBeDefined();
			expect(typeof client.generateModelResponse).toBe('function');
		});
	});

	describe('RAG Service Creation', () => {
		it('should return null when RAG indexing is disabled', () => {
			mockPlugin.settings.ragIndexing.enabled = false;

			const service = GeminiClientFactory.createRagService(mockPlugin);

			expect(service).toBeNull();
		});

		it('should return RagIndexingService when Gemini provider is selected', () => {
			mockPlugin.settings.ragIndexing.enabled = true;
			mockPlugin.settings.ragIndexing.provider = RagProvider.GEMINI;

			const service = GeminiClientFactory.createRagService(mockPlugin);

			// Should return the plugin's ragIndexing instance
			expect(service).toBe(mockPlugin.ragIndexing);
		});

		it('should return LocalRagService when Ollama provider is selected', () => {
			mockPlugin.settings.ragIndexing.enabled = true;
			mockPlugin.settings.ragIndexing.provider = RagProvider.OLLAMA;

			const service = GeminiClientFactory.createRagService(mockPlugin);

			// Should create a LocalRagService instance
			expect(service).toBeDefined();
			expect(typeof (service as any).indexVault).toBe('function');
		});
	});

	describe('Fallback Provider Resolution', () => {
		it('should provide Ollama fallback when primary is Gemini and Ollama is configured', () => {
			mockPlugin.settings.chatProvider = ModelProvider.GEMINI;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			const fallback = GeminiClientFactory['resolveFallbackProvider'](
				mockPlugin,
				ModelUseCase.CHAT,
				ModelProvider.GEMINI
			);

			expect(fallback).toBe(ModelProvider.OLLAMA);
		});

		it('should not provide fallback when primary is Gemini and Ollama not configured', () => {
			mockPlugin.settings.chatProvider = ModelProvider.GEMINI;
			mockPlugin.settings.ollama.enabled = false;

			const fallback = GeminiClientFactory['resolveFallbackProvider'](
				mockPlugin,
				ModelUseCase.CHAT,
				ModelProvider.GEMINI
			);

			expect(fallback).toBeNull();
		});

		it('should provide Gemini fallback when primary is Ollama', () => {
			mockPlugin.settings.chatProvider = ModelProvider.OLLAMA;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			const fallback = GeminiClientFactory['resolveFallbackProvider'](
				mockPlugin,
				ModelUseCase.CHAT,
				ModelProvider.OLLAMA
			);

			expect(fallback).toBe(ModelProvider.GEMINI);
		});

		it('should not provide fallback for SEARCH use case (requires tool calling)', () => {
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			const fallback = GeminiClientFactory['resolveFallbackProvider'](
				mockPlugin,
				ModelUseCase.SEARCH,
				ModelProvider.OLLAMA
			);

			expect(fallback).toBeNull();
		});

		it('should create client with fallback decorator when fallback is configured', () => {
			mockPlugin.settings.chatProvider = ModelProvider.GEMINI;
			mockPlugin.settings.ollama.enabled = true;
			mockPlugin.settings.ollama.endpoint = 'http://localhost:11434';

			const client = GeminiClientFactory.createFromPlugin(mockPlugin, ModelUseCase.CHAT);

			// Client should be wrapped with FallbackDecorator
			expect(client).toBeDefined();
			expect(typeof client.generateModelResponse).toBe('function');
			// Check if it's a FallbackDecorator by checking the class name
			expect(client.constructor.name).toBe('FallbackDecorator');
		});

		it('should create client without fallback decorator when no fallback available', () => {
			mockPlugin.settings.chatProvider = ModelProvider.GEMINI;
			mockPlugin.settings.ollama.enabled = false;

			const client = GeminiClientFactory.createFromPlugin(mockPlugin, ModelUseCase.CHAT);

			// Client should not be wrapped with FallbackDecorator
			expect(client).toBeDefined();
			expect(typeof client.generateModelResponse).toBe('function');
			// Should be RetryDecorator, not FallbackDecorator
			expect(client.constructor.name).toBe('RetryDecorator');
		});
	});
});
