/**
 * Ollama API implementation
 *
 * Implements ModelApi interface for Ollama compatibility
 */

import {
	ModelApi,
	BaseModelRequest,
	ExtendedModelRequest,
	ModelResponse,
	StreamCallback,
	StreamingModelResponse,
} from './interfaces/model-api';
import { GeminiPrompts } from '../prompts';
import type ObsidianGemini from '../main';
import { ModelCapabilities } from './types';

/**
 * Configuration for OllamaClient
 */
export interface OllamaClientConfig {
	endpoint: string; // e.g., http://localhost:11434
	model?: string;
	temperature?: number;
	topP?: number;
	streamingEnabled?: boolean;
}

/**
 * Ollama API message format
 */
interface OllamaMessage {
	role: 'system' | 'user' | 'assistant';
	content: string;
}

/**
 * Ollama API request format
 */
interface OllamaRequest {
	model: string;
	messages?: OllamaMessage[];
	prompt?: string;
	stream?: boolean;
	options?: {
		temperature?: number;
		top_p?: number;
	};
}

/**
 * Ollama API response format (non-streaming)
 */
interface OllamaResponse {
	model: string;
	message?: {
		role: string;
		content: string;
	};
	response?: string; // For /api/generate endpoint
	done: boolean;
}

/**
 * Ollama embeddings request
 */
export interface OllamaEmbeddingRequest {
	model: string;
	prompt: string;
}

/**
 * Ollama embeddings response
 */
export interface OllamaEmbeddingResponse {
	embedding: number[];
}

/**
 * OllamaClient - API wrapper for Ollama
 *
 * Implements ModelApi interface for compatibility with the plugin's architecture
 */
export class OllamaClient implements ModelApi {
	private config: OllamaClientConfig;
	private prompts: GeminiPrompts;
	private plugin?: ObsidianGemini;

	constructor(config: OllamaClientConfig, prompts?: GeminiPrompts, plugin?: ObsidianGemini) {
		this.config = {
			...config,
			temperature: config.temperature ?? 0.7,
			topP: config.topP ?? 1.0,
			streamingEnabled: config.streamingEnabled ?? true,
			endpoint: config.endpoint || 'http://100.71.158.16:11434',
		};
		this.plugin = plugin;
		this.prompts = prompts || new GeminiPrompts(plugin);
	}

	/**
	 * Get model capabilities
	 */
	getCapabilities(model?: string): ModelCapabilities {
		const modelName = (model || this.config.model || '').toLowerCase();
		const isLlava = modelName.includes('llava');

		return {
			supportsVision: isLlava,
			supportsGrounding: false, // Ollama doesn't support Google's grounding API
			supportsToolCalling: false, // Limited tool calling support, disabled for now
			supportsCloudRag: false,
			supportsLocalRag: true,
		};
	}

	/**
	 * Generate a non-streaming response
	 */
	async generateModelResponse(request: BaseModelRequest | ExtendedModelRequest): Promise<ModelResponse> {
		const ollamaRequest = this.buildOllamaRequest(request, false);

		try {
			const response = await this.fetchOllama('/api/chat', ollamaRequest);
			return this.extractModelResponse(response);
		} catch (error) {
			this.plugin?.logger.error('[OllamaClient] Error generating content:', error);
			throw error;
		}
	}

	/**
	 * Generate a streaming response
	 */
	generateStreamingResponse(
		request: BaseModelRequest | ExtendedModelRequest,
		onChunk: StreamCallback
	): StreamingModelResponse {
		let cancelled = false;
		let accumulatedText = '';

		const complete = (async (): Promise<ModelResponse> => {
			const ollamaRequest = this.buildOllamaRequest(request, true);

			try {
				const response = await fetch(`${this.config.endpoint}/api/chat`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(ollamaRequest),
				});

				if (!response.ok) {
					throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
				}

				const reader = response.body?.getReader();
				if (!reader) {
					throw new Error('No response body');
				}

				const decoder = new TextDecoder();

				while (true) {
					if (cancelled) {
						reader.cancel();
						break;
					}

					const { done, value } = await reader.read();
					if (done) break;

					const chunk = decoder.decode(value, { stream: true });
					const lines = chunk.split('\n').filter((line) => line.trim());

					for (const line of lines) {
						try {
							const json: OllamaResponse = JSON.parse(line);
							const text = json.message?.content || '';

							if (text) {
								accumulatedText += text;
								onChunk({ text });
							}

							if (json.done) {
								reader.cancel();
								return {
									markdown: accumulatedText,
									rendered: '',
								};
							}
						} catch (parseError) {
							this.plugin?.logger.warn('[OllamaClient] Failed to parse chunk:', parseError);
						}
					}
				}

				return {
					markdown: accumulatedText,
					rendered: '',
				};
			} catch (error) {
				if (cancelled) {
					return {
						markdown: accumulatedText,
						rendered: '',
					};
				}
				this.plugin?.logger.error('[OllamaClient] Streaming error:', error);
				throw error;
			}
		})();

		return {
			complete,
			cancel: () => {
				cancelled = true;
			},
		};
	}

	/**
	 * Generate embeddings for RAG
	 */
	async generateEmbedding(text: string, model?: string): Promise<number[]> {
		const embeddingModel = model || this.config.model || 'nomic-embed-text';

		const request: OllamaEmbeddingRequest = {
			model: embeddingModel,
			prompt: text,
		};

		try {
			const response = await this.fetchOllama('/api/embeddings', request);
			return response.embedding;
		} catch (error) {
			this.plugin?.logger.error('[OllamaClient] Error generating embedding:', error);
			throw error;
		}
	}

	/**
	 * Build Ollama request from our request format
	 */
	private buildOllamaRequest(request: BaseModelRequest | ExtendedModelRequest, stream: boolean): OllamaRequest {
		const isExtended = 'userMessage' in request;
		const model = request.model || this.config.model;

		if (!model) {
			throw new Error('No Ollama model configured. Please select a model in settings.');
		}

		const messages: OllamaMessage[] = [];

		if (isExtended) {
			const extReq = request as ExtendedModelRequest;

			// Add system instruction from prompt field
			if (extReq.prompt) {
				messages.push({
					role: 'system',
					content: extReq.prompt,
				});
			}

			// Add conversation history
			if (extReq.conversationHistory && extReq.conversationHistory.length > 0) {
				for (const msg of extReq.conversationHistory) {
					messages.push({
						role: msg.role === 'model' ? 'assistant' : msg.role,
						content: this.extractTextFromParts(msg.parts),
					});
				}
			}

			// Add current user message
			if (extReq.userMessage) {
				// Note: Ollama doesn't support inline attachments in the same way as Gemini
				// For now, we'll just include the text. Vision support would require llava models
				// and a different approach
				messages.push({
					role: 'user',
					content: extReq.userMessage,
				});
			}
		} else {
			// BaseModelRequest - simple prompt as user message
			const baseReq = request as BaseModelRequest;
			messages.push({
				role: 'user',
				content: baseReq.prompt,
			});
		}

		return {
			model,
			messages,
			stream,
			options: {
				temperature: request.temperature ?? this.config.temperature,
				top_p: request.topP ?? this.config.topP,
			},
		};
	}

	/**
	 * Extract text from Gemini-style parts array
	 */
	private extractTextFromParts(parts: any[]): string {
		if (!parts || !Array.isArray(parts)) return '';

		return parts
			.filter((part) => part.text)
			.map((part) => part.text)
			.join('\n');
	}

	/**
	 * Extract ModelResponse from Ollama response
	 */
	private extractModelResponse(response: OllamaResponse): ModelResponse {
		const markdown = response.message?.content || response.response || '';

		return {
			markdown,
			rendered: '', // Ollama doesn't support grounding
		};
	}

	/**
	 * Fetch from Ollama API
	 */
	private async fetchOllama(endpoint: string, body: any): Promise<any> {
		const response = await fetch(`${this.config.endpoint}${endpoint}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`Ollama API error: ${response.status} ${response.statusText} - ${errorText}`);
		}

		return response.json();
	}
}
