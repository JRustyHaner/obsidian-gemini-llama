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
	ToolCall,
} from './interfaces/model-api';
import { GeminiPrompts } from '../prompts';
import type ObsidianGemini from '../main';
import { ModelCapabilities } from './types';

/**
 * Configuration for OllamaClient
 */
export interface OllamaClientConfig {
	endpoints: Array<{
		endpoint: string;
		apiKey?: string;
		useLmStudioApi?: boolean;
		discoveredModels?: string[]; // Models discovered from this endpoint
		chatModel?: string; // Per-endpoint chat model override
		summaryModel?: string; // Per-endpoint summary model override
		completionsModel?: string; // Per-endpoint completions model override
		rewriteModel?: string; // Per-endpoint rewrite model override
	}>; // List of endpoints with optional per-endpoint API keys and LM Studio toggle
	primaryEndpointIndex?: number; // Index of primary endpoint (default: 0)
	model?: string;
	temperature?: number;
	topP?: number;
	streamingEnabled?: boolean;
	useLmStudioApi?: boolean; // If true, use LM Studio API format
	timeoutMs?: number; // Timeout in ms for non-streaming requests (default: 120000 = 120s)
	streamingTimeoutMs?: number; // Timeout in ms for streaming requests (default: 600000 = 10 min)
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
		tool_calls?: any[];
	};
	response?: string; // For /api/generate endpoint
	done: boolean;
	tool_calls?: any[];
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
	private readonly DEFAULT_TIMEOUT_MS = 120000; // 120 seconds for regular requests
	private readonly DEFAULT_STREAMING_TIMEOUT_MS = 600000; // 10 minutes for streaming

	constructor(config: OllamaClientConfig, prompts?: GeminiPrompts, plugin?: ObsidianGemini) {
		this.config = {
			...config,
			temperature: config.temperature ?? 0.7,
			topP: config.topP ?? 1.0,
			streamingEnabled: config.streamingEnabled ?? true,
			timeoutMs: config.timeoutMs ?? this.DEFAULT_TIMEOUT_MS,
			streamingTimeoutMs: config.streamingTimeoutMs ?? this.DEFAULT_STREAMING_TIMEOUT_MS,
			primaryEndpointIndex: config.primaryEndpointIndex ?? 0,
		};
		this.plugin = plugin;
		this.prompts = prompts || new GeminiPrompts(plugin);

		// Note: OllamaClient supports tools/MCP commands via system prompt descriptions
		// and best-effort extraction of tool calls from model responses
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

	async generateModelResponse(request: BaseModelRequest | ExtendedModelRequest): Promise<ModelResponse> {
		const allEndpoints = this.config.endpoints || [];

		if (allEndpoints.length === 0) {
			throw new Error('[OllamaClient] No endpoints configured');
		}

		const primaryIndex = this.config.primaryEndpointIndex ?? 0;

		// Log endpoint configuration at request start
		this.plugin?.logger.log('[OllamaClient] Non-streaming request initiated:', {
			primaryEndpointIndex: primaryIndex,
			totalEndpoints: allEndpoints.length,
			model: request.model || this.config.model,
			endpoints: allEndpoints.map((ep, idx) => ({
				index: idx,
				url: ep.endpoint,
				useLmStudio: ep.useLmStudioApi ?? false,
				apiKeyPresent: !!ep.apiKey,
			})),
		});

		// Create endpoint order: try primary first, then others
		const endpointOrder: number[] = [];
		endpointOrder.push(primaryIndex);
		for (let i = 0; i < allEndpoints.length; i++) {
			if (i !== primaryIndex) {
				endpointOrder.push(i);
			}
		}

		for (let attemptIdx = 0; attemptIdx < endpointOrder.length; attemptIdx++) {
			const endpointIdx = endpointOrder[attemptIdx];
			try {
				const endpointConfig = allEndpoints[endpointIdx];
				const ollamaRequest = await this.buildOllamaRequest(request, false, endpointConfig);

				let endpointPath = '/api/chat';
				// Per-endpoint setting takes precedence; default to false (Ollama format) if not set
				const useLmStudio = endpointConfig.useLmStudioApi === true;

				this.plugin?.logger.debug(`[OllamaClient] Endpoint ${endpointIdx} config:`, {
					endpoint: endpointConfig.endpoint,
					useLmStudioApi: endpointConfig.useLmStudioApi,
					resolved: useLmStudio,
					selectedPath: useLmStudio ? '/api/v1/chat' : '/api/chat',
					isPrimary: endpointIdx === primaryIndex,
				});

				if (useLmStudio) {
					endpointPath = '/api/v1/chat';
				}

				let fullEndpoint = endpointConfig.endpoint.trim();
				// If endpoint already ends with /api/chat or /api/v1/chat, use as is
				if (!fullEndpoint.endsWith(endpointPath)) {
					// If endpoint contains /api/chat or /api/v1/chat anywhere, do not append
					if (!fullEndpoint.match(/\/api\/(chat|v1\/chat)/)) {
						// Only append if endpoint does not already contain /api/chat or /api/v1/chat
						fullEndpoint = fullEndpoint.replace(/\/$/, '') + endpointPath;
					}
				}
				// Normalize accidental duplicated protocol or host concatenation
				const firstProto = fullEndpoint.indexOf('http');
				const lastProto = fullEndpoint.lastIndexOf('http');
				if (firstProto !== -1 && lastProto !== -1 && firstProto !== lastProto) {
					fullEndpoint = fullEndpoint.slice(lastProto);
				}

				const response = await this.fetchOllama(fullEndpoint, ollamaRequest, endpointConfig.apiKey);
				if (endpointIdx !== primaryIndex) {
					this.plugin?.logger.log(
						`[OllamaClient] Primary endpoint failed, successfully reached fallback: ${endpointConfig.endpoint}`
					);
				}

				// Log response structure before extraction
				this.plugin?.logger.log('[OllamaClient] Raw response received:', {
					hasMessage: !!response.message,
					messageKeys: response.message ? Object.keys(response.message) : [],
					fullMessage: response.message,
					messageContent: response.message?.content,
					messageThinking: (response.message as any)?.thinking?.substring(0, 100),
					responseField: response.response?.substring(0, 100),
					done: response.done,
				});

				return this.extractModelResponse(response);
			} catch (error) {
				const endpointConfig = allEndpoints[endpointIdx];
				const isLastEndpoint = attemptIdx === endpointOrder.length - 1;

				if (isLastEndpoint) {
					this.plugin?.logger.error('[OllamaClient] All endpoints failed, last error:', error);
					throw error;
				} else {
					this.plugin?.logger.warn(
						`[OllamaClient] Endpoint ${endpointIdx} (${endpointConfig.endpoint}) failed, trying next...`,
						error instanceof Error ? error.message : error
					);
				}
			}
		}

		throw new Error('[OllamaClient] No endpoints available');
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
		let parseErrorCount = 0;
		const accumulatedToolCalls: ToolCall[] = [];

		const complete = (async (): Promise<ModelResponse> => {
			// Get primary endpoint and its API key
			const primaryIndex = this.config.primaryEndpointIndex ?? 0;
			const endpoints = this.config.endpoints || [];
			if (endpoints.length === 0) {
				throw new Error('[OllamaClient] No endpoints configured');
			}
			const endpointConfig = endpoints[primaryIndex];
			const ollamaRequest = await this.buildOllamaRequest(request, true, endpointConfig);

			// Debug logging - log at request initiation
			this.plugin?.logger.log('[OllamaClient] Streaming request initiated:', {
				primaryEndpointIndex: primaryIndex,
				primaryEndpointUrl: endpointConfig.endpoint,
				primaryEndpointUseLmStudio: endpointConfig.useLmStudioApi ?? false,
				model: request.model || this.config.model,
				totalEndpoints: endpoints.length,
				endpoints: endpoints.map((ep, idx) => ({
					index: idx,
					url: ep.endpoint,
					useLmStudio: ep.useLmStudioApi ?? false,
				})),
			});

			let endpointPath = '/api/chat';
			// Per-endpoint setting takes precedence; default to false (Ollama format) if not set
			const useLmStudio = endpointConfig.useLmStudioApi === true;

			this.plugin?.logger.debug(`[OllamaClient] Streaming - Primary endpoint ${primaryIndex} config:`, {
				endpoint: endpointConfig.endpoint,
				useLmStudioApi: endpointConfig.useLmStudioApi,
				resolved: useLmStudio,
				selectedPath: useLmStudio ? '/api/v1/chat' : '/api/chat',
			});

			if (useLmStudio) {
				endpointPath = '/api/v1/chat';
			}
			let fullEndpoint = endpointConfig.endpoint.trim();
			// If endpoint already ends with /api/chat or /api/v1/chat, use as is
			if (!fullEndpoint.endsWith(endpointPath)) {
				// If endpoint contains /api/chat or /api/v1/chat anywhere, do not append
				if (!fullEndpoint.match(/\/api\/(chat|v1\/chat)/)) {
					fullEndpoint = fullEndpoint.replace(/\/$/, '') + endpointPath;
				}
			}
			// Normalize accidental duplicated protocol or host concatenation
			const fp = fullEndpoint.indexOf('http');
			const lp = fullEndpoint.lastIndexOf('http');
			if (fp !== -1 && lp !== -1 && fp !== lp) {
				fullEndpoint = fullEndpoint.slice(lp);
			}
			this.plugin?.logger.debug('[OllamaClient] Request:', {
				endpoint: fullEndpoint,
				model: ollamaRequest.model,
				messageCount: ollamaRequest.messages?.length,
			});

			try {
				const headers: Record<string, string> = { 'Content-Type': 'application/json' };
				if (endpointConfig.apiKey) {
					headers['Authorization'] = `Bearer ${endpointConfig.apiKey}`;
				}

				const abortController = new AbortController();
				const timeoutId = setTimeout(() => abortController.abort(), this.config.streamingTimeoutMs);

				const response = await fetch(fullEndpoint, {
					method: 'POST',
					headers,
					body: JSON.stringify(ollamaRequest),
					signal: abortController.signal,
				}).finally(() => clearTimeout(timeoutId));

				if (!response.ok) {
					throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
				}

				const reader = response.body?.getReader();
				if (!reader) {
					throw new Error('No response body');
				}

				const decoder = new TextDecoder();
				let buffer = '';

				const processPayload = (payload: string): boolean => {
					const trimmed = payload.trim();
					if (!trimmed) return false;

					if (trimmed === '[DONE]') {
						return true;
					}

					try {
						const json = JSON.parse(trimmed) as any;
						const parsedToolCalls = this.extractToolCallsFromPayload(json);
						if (parsedToolCalls && parsedToolCalls.length > 0) {
							accumulatedToolCalls.push(...parsedToolCalls);
						}
						const text =
							json?.message?.content ??
							json?.message?.delta ??
							json?.message?.thinking ?? // For qwen models with extended thinking
							json?.response ??
							json?.content ??
							json?.delta ??
							json?.thinking ?? // For root-level thinking field
							json?.delta?.content ??
							json?.data?.content ??
							json?.data?.delta ??
							json?.choices?.[0]?.delta?.content ??
							'';

						if (text) {
							const filtered = this.filterThinkingContent(text);
							if (!filtered && text) {
								this.plugin?.logger.debug('[OllamaClient] Text extracted but filtered to empty:', {
									original: text.substring(0, 100),
									filtered: filtered.substring(0, 100),
									hadThinking: text.includes('thinking') || text.includes('think'),
								});
							}
							accumulatedText += filtered;
							if (filtered) onChunk({ text: filtered });
						}

						if (json?.done === true) {
							return true;
						}
					} catch (parseError) {
						// Avoid flooding logs on SSE/non-JSON control lines.
						parseErrorCount += 1;
						if (parseErrorCount <= 3) {
							this.plugin?.logger.warn('[OllamaClient] Failed to parse chunk:', parseError);
						}
					}

					return false;
				};

				while (true) {
					if (cancelled) {
						reader.cancel();
						break;
					}

					const { done, value } = await reader.read().catch((error) => {
						if (error instanceof Error && error.name === 'AbortError') {
							throw new Error(`Ollama streaming request timeout after ${this.config.streamingTimeoutMs}ms`);
						}
						throw error;
					});
					if (done) {
						break;
					}

					buffer += decoder.decode(value, { stream: true });

					let newlineIndex = buffer.indexOf('\n');
					while (newlineIndex !== -1) {
						const rawLine = buffer.slice(0, newlineIndex);
						buffer = buffer.slice(newlineIndex + 1);
						const line = rawLine.trim();

						if (!line || line.startsWith('event:')) {
							newlineIndex = buffer.indexOf('\n');
							continue;
						}

						const payload = line.startsWith('data:') ? line.slice(5).trim() : line;
						const isDone = processPayload(payload);
						if (isDone) {
							reader.cancel();
							return {
								markdown: accumulatedText,
								rendered: '',
								...(accumulatedToolCalls.length > 0 ? { toolCalls: accumulatedToolCalls } : {}),
							};
						}

						newlineIndex = buffer.indexOf('\n');
					}
				}

				if (buffer.trim()) {
					const trailing = buffer.trim();
					if (!trailing.startsWith('event:')) {
						const payload = trailing.startsWith('data:') ? trailing.slice(5).trim() : trailing;
						processPayload(payload);
					}
				}

				// Some OpenAI-compatible backends may stream SSE metadata without
				// content in a shape we don't decode. Try a final non-stream request
				// once before returning an empty result.
				if (!cancelled && !accumulatedText.trim()) {
					this.plugin?.logger.warn('[OllamaClient] Streaming returned empty text, attempting non-stream fallback');
					try {
						const nonStream = await this.generateModelResponse(request);
						this.plugin?.logger.log('[OllamaClient] Non-stream fallback result:', {
							hasMarkdown: !!nonStream.markdown,
							markdownLength: nonStream.markdown?.length || 0,
							markdownPreview: nonStream.markdown?.substring(0, 100),
						});
						if (nonStream.markdown?.trim()) {
							accumulatedText = nonStream.markdown;
						}
					} catch (fallbackError) {
						this.plugin?.logger.warn('[OllamaClient] Non-stream fallback after empty stream failed:', fallbackError);
					}
				}

				return {
					markdown: accumulatedText,
					rendered: '',
					...(accumulatedToolCalls.length > 0 ? { toolCalls: accumulatedToolCalls } : {}),
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
			// Use primary endpoint for embeddings
			const primaryIndex = this.config.primaryEndpointIndex ?? 0;
			const endpoints = this.config.endpoints || [];
			const endpointConfig = endpoints[primaryIndex];
			if (!endpointConfig) {
				throw new Error('[OllamaClient] No primary endpoint configured');
			}
			const response = await this.fetchOllama('/api/embeddings', request, endpointConfig.apiKey);
			return response.embedding;
		} catch (error) {
			this.plugin?.logger.error('[OllamaClient] Error generating embedding:', error);
			throw error;
		}
	}

	/**
	 * Build Ollama request from our request format
	 */
	private async buildOllamaRequest(
		request: BaseModelRequest | ExtendedModelRequest,
		stream: boolean,
		endpointConfig?: any
	): Promise<OllamaRequest> {
		const isExtended = 'userMessage' in request;
		const model = request.model || this.config.model;

		if (!model) {
			throw new Error('No Ollama model configured. Please select a model in settings.');
		}

		const messages: OllamaMessage[] = [];

		if (isExtended) {
			const extReq = request as ExtendedModelRequest;
			const systemInstruction = await this.buildSystemInstruction(extReq);

			// Add unified system instruction (same pipeline as Gemini client)
			if (systemInstruction) {
				messages.push({
					role: 'system',
					content: systemInstruction,
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
				messages.push({
					role: 'user',
					content: extReq.userMessage,
				});
			}

			this.plugin?.logger.log('[OllamaClient] Built ExtendedModelRequest:', {
				isExtended,
				userMessage: extReq.userMessage?.substring(0, 100),
				historyLength: extReq.conversationHistory?.length,
				messagesCount: messages.length,
			});
		} else {
			// BaseModelRequest - simple prompt as user message
			const baseReq = request as BaseModelRequest;
			this.plugin?.logger.log('[OllamaClient] Built BaseModelRequest:', {
				isExtended,
				prompt: baseReq.prompt?.substring(0, 100),
			});
			messages.push({
				role: 'user',
				content: baseReq.prompt,
			});
		}

		// Check per-endpoint LM Studio setting - endpoint config takes precedence, defaults to false (Ollama format)
		const useLmStudio = endpointConfig?.useLmStudioApi === true;

		if (useLmStudio) {
			// LM Studio / OpenAI-compatible endpoints commonly require an
			// `input` string instead of a `messages` array. Send only `input`
			// and an `options` block to avoid "Unrecognized key(s): 'messages'"
			// errors on some deployments.
			const inputString = messages.map((m) => m.content).join('\n\n');
			return {
				model,
				input: inputString,
				stream,
				temperature: request.temperature ?? this.config.temperature,
				top_p: request.topP ?? this.config.topP,
			} as any;
		}

		// Ollama format
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
	 * Build unified system instruction for Ollama requests.
	 * Mirrors GeminiClient behavior: AGENTS.md + skills + tool instructions + custom prompt.
	 *
	 * **Tool/MCP Support for Ollama:**
	 * - Ollama models have access to available tools through the system prompt
	 * - Tools and MCP commands are described in natural language in the system message
	 * - The model can request tool execution via function calls (if the backend supports it)
	 * - Tool descriptions, parameters, and usage instructions are included in the system prompt
	 * - For Ollama to execute tools, the response must include tool call syntax (e.g., `{"name": "tool", "arguments": {...}}`)
	 * - Tool extraction is implemented for OpenAI-compatible tool_calls format
	 */
	private async buildSystemInstruction(extReq: ExtendedModelRequest): Promise<string> {
		let agentsMemory: string | null = null;
		if (this.plugin?.agentsMemory) {
			try {
				agentsMemory = await this.plugin.agentsMemory.read();
			} catch (error) {
				this.plugin.logger.warn('Failed to load AGENTS.md:', error);
			}
		}

		let availableSkills: { name: string; description: string }[] = [];
		if (this.plugin?.skillManager) {
			try {
				availableSkills = await this.plugin.skillManager.getSkillSummaries();
			} catch (error) {
				this.plugin.logger.warn('Failed to load skill summaries:', error);
			}
		}

		let systemInstruction = this.prompts.getSystemPromptWithCustom(
			extReq.availableTools,
			extReq.customPrompt,
			agentsMemory,
			availableSkills
		);

		if (extReq.prompt && !extReq.customPrompt?.overrideSystemPrompt) {
			systemInstruction += '\n\n' + extReq.prompt;
		}

		return systemInstruction;
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
	 * Remove <think>...</think> tags and other thinking markers from text
	 * Some models like deepseek-r1 wrap reasoning in these tags
	 */
	private filterThinkingContent(text: string): string {
		if (!text) return text;

		// Remove <think>...</think> tags and their content (supports multi-line)
		let filtered = text.replace(/<think>[\s\S]*?<\/think>/g, '');

		// Remove other common thinking markers (some models might use these)
		filtered = filtered.replace(/<thinking>[\s\S]*?<\/thinking>/g, '');
		filtered = filtered.replace(/\[thinking\][\s\S]*?\[\/thinking\]/g, '');

		// Only clean up extra whitespace if we actually removed something
		// (avoid collapsing intentional spaces in the original text)
		if (filtered !== text) {
			filtered = filtered.replace(/\n\s*\n/g, '\n');
		}

		return filtered;
	}

	/**
	 * Extract ModelResponse from Ollama response
	 */
	private extractModelResponse(response: OllamaResponse): ModelResponse {
		let markdown = response.message?.content || (response.message as any)?.thinking || response.response || '';
		this.plugin?.logger.debug('[OllamaClient] Response extraction:', {
			hasContent: !!response.message?.content,
			hasThinking: !!(response.message as any)?.thinking,
			contentLength: response.message?.content?.length || 0,
			thinkingLength: (response.message as any)?.thinking?.length || 0,
			selectedSource: response.message?.content
				? 'content'
				: (response.message as any)?.thinking
					? 'thinking'
					: response.response
						? 'response'
						: 'empty',
		});
		markdown = this.filterThinkingContent(markdown);
		const toolCalls = this.extractToolCallsFromPayload(response);

		return {
			markdown,
			rendered: '', // Ollama doesn't support grounding
			...(toolCalls && toolCalls.length > 0 ? { toolCalls } : {}),
		};
	}

	/**
	 * Best-effort extraction of tool calls from Ollama/OpenAI-compatible payloads
	 */
	private extractToolCallsFromPayload(payload: any): ToolCall[] | undefined {
		const rawToolCalls =
			payload?.message?.tool_calls ||
			payload?.tool_calls ||
			payload?.choices?.[0]?.message?.tool_calls ||
			payload?.choices?.[0]?.delta?.tool_calls;

		if (!Array.isArray(rawToolCalls) || rawToolCalls.length === 0) {
			return undefined;
		}

		const calls: ToolCall[] = [];
		for (const call of rawToolCalls) {
			const name = call?.function?.name || call?.name;
			const argsRaw = call?.function?.arguments ?? call?.arguments ?? {};
			if (!name) continue;

			let args: Record<string, any> = {};
			if (typeof argsRaw === 'string') {
				try {
					args = JSON.parse(argsRaw);
				} catch {
					// Streaming deltas may provide partial argument strings
					continue;
				}
			} else if (argsRaw && typeof argsRaw === 'object') {
				args = argsRaw;
			}

			calls.push({
				name,
				arguments: args,
			});
		}

		return calls.length > 0 ? calls : undefined;
	}

	/**
	 * Fetch from Ollama API
	 */
	private async fetchOllama(endpoint: string, body: any, apiKey?: string): Promise<any> {
		// Log the request being sent
		this.plugin?.logger.log('[OllamaClient] Sending request:', {
			endpoint,
			model: body.model,
			hasMessages: !!body.messages,
			hasInput: !!body.input,
			messageCount: body.messages?.length,
			firstMessage: body.messages?.[0],
			stream: body.stream,
			temperature: body.options?.temperature || body.temperature,
			bodyKeys: Object.keys(body),
		});

		const headers: Record<string, string> = { 'Content-Type': 'application/json' };
		if (apiKey) {
			headers['Authorization'] = `Bearer ${apiKey}`;
		}

		const url = endpoint.startsWith('http') ? endpoint : endpoint;
		const doFetch = async (bdy: any) => {
			const abortController = new AbortController();
			const timeoutId = setTimeout(() => abortController.abort(), this.config.timeoutMs);

			return fetch(url, {
				method: 'POST',
				headers,
				body: JSON.stringify(bdy),
				signal: abortController.signal,
			})
				.catch((error) => {
					clearTimeout(timeoutId);
					if (error instanceof Error && error.name === 'AbortError') {
						throw new Error(`Request timeout: Ollama/LM Studio did not respond within ${this.config.timeoutMs}ms`);
					}
					throw error;
				})
				.finally(() => clearTimeout(timeoutId));
		};

		let response = await doFetch(body);

		if (!response.ok) {
			const errorText = await response.text();
			this.plugin?.logger.debug('[OllamaClient] Initial error from server:', errorText);

			// Common compatibility errors - try a single adaptive retry with transformed payload
			const needsOptionsUnwrap = /Unrecognized key\(s\) in object: 'options'/.test(errorText);
			const inputRequired = /'input' is required/.test(errorText) || /"input" is required/.test(errorText);
			const messagesUnrecognized = /Unrecognized key\(s\) in object: 'messages'/.test(errorText);

			if (needsOptionsUnwrap || inputRequired || messagesUnrecognized) {
				this.plugin?.logger.warn('[OllamaClient] Detected incompatible payload schema, attempting compatibility retry');

				// Build an adapted body
				const alt = JSON.parse(JSON.stringify(body));

				// If server dislikes `options`, move its fields to top-level
				if (alt.options) {
					alt.temperature = alt.temperature ?? alt.options.temperature;
					alt.top_p = alt.top_p ?? alt.options.top_p;
					delete alt.options;
				}

				// If server requires `input`, convert `messages` -> `input`
				if ((inputRequired || messagesUnrecognized) && alt.messages && Array.isArray(alt.messages)) {
					alt.input = alt.messages.map((m: any) => m.content || '').join('\n\n');
					delete alt.messages;
				}

				// If server complains about messages but we only have `input`, try to build `messages`
				if (messagesUnrecognized && alt.input && !alt.messages) {
					alt.messages = [{ role: 'user', content: alt.input }];
					delete alt.input;
				}

				this.plugin?.logger.debug('[OllamaClient] Retry payload:', alt);

				response = await doFetch(alt);
				if (!response.ok) {
					const retryErr = await response.text();
					throw new Error(`Ollama API error: ${response.status} ${response.statusText} - ${retryErr}`);
				}
			} else {
				throw new Error(`Ollama API error: ${response.status} ${response.statusText} - ${errorText}`);
			}
		}

		try {
			return response.json();
		} catch (error) {
			if (error instanceof Error && error.name === 'AbortError') {
				throw new Error(`Request timeout: Ollama/LM Studio did not respond within ${this.config.timeoutMs}ms`);
			}
			throw error;
		}
	}
}
