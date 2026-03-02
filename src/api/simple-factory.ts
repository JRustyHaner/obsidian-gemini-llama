/**
 * Simplified factory for creating API clients
 *
 * Supports multiple providers (Gemini, Ollama) with intelligent routing based on
 * user settings and feature requirements. Handles fallback chains for resilience.
 */

import { GeminiClient, GeminiClientConfig } from './gemini-client';
import { OllamaClient, OllamaClientConfig } from './ollama-client';
import { ModelApi } from './interfaces/model-api';
import { ModelProvider, RagProvider } from './types';
import { GeminiPrompts } from '../prompts';
import { RetryDecorator } from './retry-decorator';
import { FallbackDecorator } from './fallback-decorator';
import { LocalRagService } from '../services/local-rag-service';
import { RagIndexingService } from '../services/rag-indexing';
import { getDefaultModelForRole } from '../models';
import type ObsidianGemini from '../main';

/**
 * Model use cases for the plugin
 */
export enum ModelUseCase {
	CHAT = 'chat',
	SUMMARY = 'summary',
	COMPLETIONS = 'completions',
	REWRITE = 'rewrite',
	SEARCH = 'search',
}

/**
 * Simple factory for creating Gemini API clients
 */
export class GeminiClientFactory {
	/**
	 * Create a model API client from plugin settings
	 *
	 * Intelligently routes to the appropriate provider (Gemini or Ollama) based on
	 * user settings and feature requirements. Falls back to Gemini if the primary
	 * provider is not configured or unavailable.
	 *
	 * If a fallback provider is configured, wraps the primary client with
	 * FallbackDecorator to automatically switch to fallback on runtime errors.
	 *
	 * @param plugin - Plugin instance with settings
	 * @param useCase - The use case for this model (determines which model to use)
	 * @param overrides - Optional config overrides (for per-session settings)
	 * @returns Configured ModelApi instance
	 */
	static createFromPlugin(
		plugin: ObsidianGemini,
		useCase: ModelUseCase,
		overrides?: Partial<GeminiClientConfig>
	): ModelApi {
		const settings = plugin.settings;

		// Determine which provider to use based on settings and use case
		const provider = this.resolveProvider(plugin, useCase);
		const fallbackProvider = this.resolveFallbackProvider(plugin, useCase, provider);

		// Create primary client
		let primaryClient: ModelApi;
		if (provider === ModelProvider.OLLAMA) {
			primaryClient = this.createOllamaClient(plugin, useCase, overrides);
		} else {
			primaryClient = this.createGeminiClient(plugin, useCase, overrides);
		}

		// If a fallback provider is configured, wrap with fallback decorator
		if (fallbackProvider) {
			let fallbackClient: ModelApi;
			if (fallbackProvider === ModelProvider.OLLAMA) {
				fallbackClient = this.createOllamaClient(plugin, useCase, overrides);
			} else {
				fallbackClient = this.createGeminiClient(plugin, useCase, overrides);
			}

			return new FallbackDecorator({
				primary: primaryClient,
				fallback: fallbackClient,
				logger: plugin.logger,
			});
		}

		// No fallback configured, return primary client as-is
		return primaryClient;
	}

	/**
	 * Create a Gemini API client
	 *
	 * @param plugin - Plugin instance with settings
	 * @param useCase - The use case for this model
	 * @param overrides - Optional config overrides
	 * @returns Configured GeminiClient instance wrapped with retry logic
	 */
	private static createGeminiClient(
		plugin: ObsidianGemini,
		useCase: ModelUseCase,
		overrides?: Partial<GeminiClientConfig>
	): ModelApi {
		const settings = plugin.settings;

		// Determine which model to use based on use case
		let modelName: string;
		switch (useCase) {
			case ModelUseCase.CHAT:
				modelName = settings.chatModelName || getDefaultModelForRole('chat');
				break;
			case ModelUseCase.SUMMARY:
				modelName = settings.summaryModelName || getDefaultModelForRole('summary');
				break;
			case ModelUseCase.COMPLETIONS:
				modelName = settings.completionsModelName || getDefaultModelForRole('completions');
				break;
			case ModelUseCase.REWRITE:
				// Rewrite uses chat model
				modelName = settings.chatModelName || getDefaultModelForRole('chat');
				break;
			case ModelUseCase.SEARCH:
				// Search uses chat model
				modelName = settings.chatModelName || getDefaultModelForRole('chat');
				break;
			default:
				modelName = getDefaultModelForRole('chat');
		}

		// Build config
		const config: GeminiClientConfig = {
			apiKey: plugin.apiKey,
			model: modelName,
			temperature: settings.temperature ?? 1.0,
			topP: settings.topP ?? 0.95,
			streamingEnabled: settings.streamingEnabled ?? true,
			...overrides,
		};

		// Create prompts instance with plugin reference so it can access settings
		const prompts = new GeminiPrompts(plugin);

		// Create client
		const client = new GeminiClient(config, prompts, plugin);

		// Wrap with retry decorator
		const retryConfig = {
			maxRetries: settings.maxRetries ?? 3,
			initialBackoffDelay: settings.initialBackoffDelay ?? 1000,
		};

		return new RetryDecorator(client, retryConfig, plugin.logger);
	}

	/**
	 * Create an Ollama API client
	 *
	 * @param plugin - Plugin instance with settings
	 * @param useCase - The use case for this model
	 * @param overrides - Optional config overrides
	 * @returns Configured OllamaClient instance wrapped with retry logic
	 */
	private static createOllamaClient(
		plugin: ObsidianGemini,
		useCase: ModelUseCase,
		overrides?: Partial<GeminiClientConfig>
	): ModelApi {
		const settings = plugin.settings;
		const ollamaSettings = settings.ollama;

		if (!ollamaSettings.enabled) {
			plugin.logger.warn('Ollama selected but not enabled, falling back to Gemini');
			return this.createGeminiClient(plugin, useCase, overrides);
		}

		// Determine which model to use based on use case
		let modelName: string;
		switch (useCase) {
			case ModelUseCase.CHAT:
				modelName = ollamaSettings.models.chat || getDefaultModelForRole('chat');
				break;
			case ModelUseCase.SUMMARY:
				modelName = ollamaSettings.models.summary || getDefaultModelForRole('summary');
				break;
			case ModelUseCase.COMPLETIONS:
				modelName = ollamaSettings.models.completions || getDefaultModelForRole('completions');
				break;
			case ModelUseCase.REWRITE:
				modelName = ollamaSettings.models.rewrite || getDefaultModelForRole('chat');
				break;
			case ModelUseCase.SEARCH:
				modelName = ollamaSettings.models.chat || getDefaultModelForRole('chat');
				break;
			default:
				modelName = getDefaultModelForRole('chat');
		}

		// Build config for Ollama
		const config: OllamaClientConfig = {
			endpoint: ollamaSettings.endpoint,
			model: modelName,
			temperature: settings.temperature ?? 0.7,
			topP: settings.topP ?? 0.95,
			streamingEnabled: settings.streamingEnabled ?? true,
		};

		// Create prompts and client
		const prompts = new GeminiPrompts(plugin);
		const client = new OllamaClient(config, prompts, plugin);

		// Wrap with retry decorator
		const retryConfig = {
			maxRetries: settings.maxRetries ?? 3,
			initialBackoffDelay: settings.initialBackoffDelay ?? 1000,
		};

		return new RetryDecorator(client, retryConfig, plugin.logger);
	}

	/**
	 * Resolve which provider to use for a given use case
	 *
	 * Follows user preference order with fallback to Gemini for safety.
	 * Some features like tools always prefer Gemini unless fallback fails.
	 *
	 * @param plugin - Plugin instance
	 * @param useCase - The use case for this model
	 * @returns The provider to use (ModelProvider enum)
	 */
	private static resolveProvider(plugin: ObsidianGemini, useCase: ModelUseCase): ModelProvider {
		const settings = plugin.settings;

		// Determine provider based on use case
		let selectedProvider: ModelProvider;
		switch (useCase) {
			case ModelUseCase.CHAT:
				selectedProvider = settings.chatProvider ?? ModelProvider.GEMINI;
				break;
			case ModelUseCase.SUMMARY:
				selectedProvider = settings.summaryProvider ?? ModelProvider.GEMINI;
				break;
			case ModelUseCase.COMPLETIONS:
				selectedProvider = settings.completionsProvider ?? ModelProvider.GEMINI;
				break;
			case ModelUseCase.REWRITE:
				selectedProvider = settings.rewriteProvider ?? ModelProvider.GEMINI;
				break;
			// Tools and search always use Gemini (they need tool calling and grounding)
			case ModelUseCase.SEARCH:
				return ModelProvider.GEMINI;
			default:
				return ModelProvider.GEMINI;
		}

		// Validate that the selected provider is properly configured
		if (selectedProvider === ModelProvider.OLLAMA) {
			if (!settings.ollama.enabled) {
				plugin.logger.warn(`Ollama selected for ${useCase} but not enabled, falling back to Gemini`);
				return ModelProvider.GEMINI;
			}
			if (!settings.ollama.endpoint) {
				plugin.logger.warn(`Ollama selected for ${useCase} but endpoint not configured, falling back to Gemini`);
				return ModelProvider.GEMINI;
			}
		}

		return selectedProvider;
	}

	/**
	 * Resolve the fallback provider for a given use case
	 *
	 * Returns a fallback provider if configured and different from primary.
	 * This enables automatic provider switching on runtime failures.
	 *
	 * Strategy:
	 * - If primary is Gemini, fallback is Ollama (if enabled and configured)
	 * - If primary is Ollama, fallback is Gemini (always available)
	 * - Returns null if fallback is same as primary or not properly configured
	 *
	 * @param plugin - Plugin instance
	 * @param useCase - The use case for this model
	 * @param primaryProvider - The resolved primary provider
	 * @returns The fallback provider, or null if no fallback
	 */
	private static resolveFallbackProvider(
		plugin: ObsidianGemini,
		useCase: ModelUseCase,
		primaryProvider: ModelProvider
	): ModelProvider | null {
		const settings = plugin.settings;

		// Tools and search always use Gemini, no fallback needed
		if (useCase === ModelUseCase.SEARCH) {
			return null;
		}

		// If primary is Gemini, fallback is Ollama (if available)
		if (primaryProvider === ModelProvider.GEMINI) {
			if (settings.ollama.enabled && settings.ollama.endpoint) {
				return ModelProvider.OLLAMA;
			}
			return null; // No fallback available
		}

		// If primary is Ollama, fallback is always Gemini
		if (primaryProvider === ModelProvider.OLLAMA) {
			return ModelProvider.GEMINI;
		}

		return null;
	}

	/**
	 * Create a GeminiClient with custom configuration
	 *
	 * @param config - Complete client configuration
	 * @param prompts - Optional prompts instance
	 * @param plugin - Optional plugin instance
	 * @returns Configured GeminiClient instance wrapped with retry logic
	 */
	static createCustom(config: GeminiClientConfig, prompts?: GeminiPrompts, plugin?: ObsidianGemini): ModelApi {
		const client = new GeminiClient(config, prompts, plugin);

		// Use retry config from plugin settings if available, otherwise use defaults
		const retryConfig = plugin
			? {
					maxRetries: plugin.settings.maxRetries ?? 3,
					initialBackoffDelay: plugin.settings.initialBackoffDelay ?? 1000,
				}
			: {
					maxRetries: 3,
					initialBackoffDelay: 1000,
				};

		return new RetryDecorator(client, retryConfig, plugin?.logger);
	}

	/**
	 * Create a chat model with optional session-specific overrides
	 *
	 * @param plugin - Plugin instance
	 * @param sessionConfig - Optional session-level config (model, temperature, topP)
	 * @returns Configured GeminiClient for chat
	 */
	static createChatModel(
		plugin: ObsidianGemini,
		sessionConfig?: { model?: string; temperature?: number; topP?: number }
	): ModelApi {
		const overrides: Partial<GeminiClientConfig> = {};

		if (sessionConfig) {
			// Session config takes precedence
			if (sessionConfig.temperature !== undefined) {
				overrides.temperature = sessionConfig.temperature;
			}
			if (sessionConfig.topP !== undefined) {
				overrides.topP = sessionConfig.topP;
			}
			// Note: model override is handled at request time via session.modelConfig
		}

		return this.createFromPlugin(plugin, ModelUseCase.CHAT, overrides);
	}

	/**
	 * Create a summary model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured GeminiClient for summaries
	 */
	static createSummaryModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.SUMMARY);
	}

	/**
	 * Create a completions model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured GeminiClient for completions
	 */
	static createCompletionsModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.COMPLETIONS);
	}

	/**
	 * Create a rewrite model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured GeminiClient for rewriting
	 */
	static createRewriteModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.REWRITE);
	}

	/**
	 * Create a search model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured GeminiClient for search operations
	 */
	static createSearchModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.SEARCH);
	}

	/**
	 * Create a RAG service instance
	 *
	 * Returns the appropriate RAG service based on user settings:
	 * - RagIndexingService for Gemini File Search
	 * - LocalRagService for Ollama local embeddings
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured RAG service instance
	 */
	static createRagService(plugin: ObsidianGemini): RagIndexingService | LocalRagService | null {
		const settings = plugin.settings;

		if (!settings.ragIndexing.enabled) {
			return null;
		}

		// Determine which RAG provider to use
		const ragProvider = settings.ragIndexing.provider ?? RagProvider.GEMINI;

		if (ragProvider === RagProvider.OLLAMA) {
			// Return LocalRagService for Ollama embeddings
			return new LocalRagService(plugin);
		}

		// Default to RagIndexingService for Gemini File Search
		// This is already created and managed by the plugin
		return plugin.ragIndexing;
	}
}
