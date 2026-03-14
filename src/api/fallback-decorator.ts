/**
 * Fallback Decorator for ModelApi
 *
 * Wraps a primary ModelApi client with a fallback provider.
 * If the primary client fails with a network or API error,
 * automatically retries with the fallback client.
 * Improves resilience by providing automatic provider switching.
 */

import {
	ModelApi,
	ModelResponse,
	BaseModelRequest,
	ExtendedModelRequest,
	StreamCallback,
	StreamingModelResponse,
} from './interfaces/model-api';
import type ObsidianGemini from '../main';

/**
 * Represents a fallback chain configuration
 */
export interface FallbackConfig {
	primary: ModelApi;
	fallback: ModelApi | null;
	logger?: any;
	plugin?: ObsidianGemini | null;
	primaryName?: string;
	fallbackName?: string;
}

/**
 * Error classification for determining fallback behavior
 */
enum ErrorType {
	NETWORK = 'network',
	API = 'api',
	AUTHENTICATION = 'authentication',
	RATE_LIMIT = 'rate_limit',
	MODEL_NOT_FOUND = 'model_not_found',
	OTHER = 'other',
}

/**
 * Fallback Decorator that wraps a ModelApi with automatic provider switching
 */
export class FallbackDecorator implements ModelApi {
	private primary: ModelApi;
	private fallback: ModelApi | null;
	private logger?: any;
	private plugin?: ObsidianGemini | null;
	private primaryName: string;
	private fallbackName: string;

	constructor(config: FallbackConfig) {
		this.primary = config.primary;
		this.fallback = config.fallback;
		this.logger = config.logger;
		this.plugin = config.plugin ?? null;
		this.primaryName = config.primaryName || 'primary';
		this.fallbackName = config.fallbackName || 'fallback';
	}

	/**
	 * Generate a model response, falling back to alternate provider on failure
	 */
	async generateModelResponse(request: BaseModelRequest | ExtendedModelRequest): Promise<ModelResponse> {
		try {
			return await this.primary.generateModelResponse(request);
		} catch (error) {
			if (!this.fallback) {
				throw error;
			}

			const errorType = this.classifyError(error);

			// Only fallback on transient errors (network, rate limit, API errors)
			// Don't fallback on authentication or model not found errors
			if (errorType === ErrorType.AUTHENTICATION || errorType === ErrorType.MODEL_NOT_FOUND) {
				throw error;
			}

			this.logger?.warn(
				`Primary provider failed (${errorType}), attempting fallback...`,
				error instanceof Error ? error.message : String(error)
			);

			// Only allow a single fallback attempt per request
			const alreadyFellBack = (request as any)?.__fallbackAttempted === true;
			if (alreadyFellBack) {
				// Already attempted fallback for this request - rethrow original error
				throw error;
			}
			// Mark this request as having attempted fallback
			try {
				(request as any).__fallbackAttempted = true;
			} catch (e) {
				// ignore
			}

			// Show fallback as a tool-execution style message in Agent UI if available
			try {
				const errMsg = error instanceof Error ? error.message : String(error);
				const execId = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
				try {
					const modelName = (request as any)?.model || '';
					const message = `Falling back to ${this.fallbackName} : ${modelName || 'unknown model'}`;
					if (
						this.plugin &&
						(this.plugin as any).agentView &&
						typeof (this.plugin as any).agentView.showToolExecution === 'function'
					) {
						// @ts-ignore
						await (this.plugin as any).agentView.showToolExecution(
							'provider-fallback',
							{
								primary: this.primaryName,
								fallback: this.fallbackName,
								model: modelName,
								message,
								error: errMsg,
							},
							execId
						);
					}
				} catch (e) {
					// ignore UI errors
				}
			} catch (e) {
				// ignore UI errors
			}

			try {
				const resp = await this.fallback.generateModelResponse(request);
				// If agent view tool execution was shown, mark it completed
				try {
					try {
						const execId2 = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
						if (
							this.plugin &&
							(this.plugin as any).agentView &&
							typeof (this.plugin as any).agentView.showToolResult === 'function'
						) {
							// @ts-ignore
							await (this.plugin as any).agentView.showToolResult(
								'provider-fallback',
								{
									success: true,
									data: {
										primary: this.primaryName,
										fallback: this.fallbackName,
										model: (request as any)?.model || '',
									},
								},
								execId2
							);
						}
					} catch (e) {
						// ignore
					}
				} catch (e) {
					// ignore
				}
				return resp;
			} catch (fallbackError) {
				// If fallback also fails, show failure in agent UI then throw original error
				this.logger?.error('Fallback provider also failed', fallbackError);
				try {
					const execId = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
					if (
						this.plugin &&
						(this.plugin as any).agentView &&
						typeof (this.plugin as any).agentView.showToolResult === 'function'
					) {
						// @ts-ignore
						await (this.plugin as any).agentView.showToolResult(
							'provider-fallback',
							{ success: false, error: String(fallbackError) },
							execId
						);
					}
				} catch (e) {
					// ignore
				}
				throw error;
			}
		}
	}

	/**
	 * Generate a streaming model response, falling back on failure
	 */
	generateStreamingResponse(
		request: BaseModelRequest | ExtendedModelRequest,
		onChunk: StreamCallback
	): StreamingModelResponse {
		// Start the streaming operation, but handle fallback asynchronously
		let cancelled = false;

		const complete = this.performStreamingWithFallback(request, onChunk).catch((error) => {
			throw error;
		});

		return {
			complete,
			cancel: () => {
				cancelled = true;
			},
		};
	}

	/**
	 * Perform streaming with fallback support
	 */
	private async performStreamingWithFallback(
		request: BaseModelRequest | ExtendedModelRequest,
		onChunk: StreamCallback
	): Promise<ModelResponse> {
		try {
			const response = this.primary.generateStreamingResponse?.(request, onChunk);
			if (!response) {
				// Primary doesn't support streaming, fall back to non-streaming
				return await this.primary.generateModelResponse(request);
			}
			return await response.complete;
		} catch (error) {
			if (!this.fallback) {
				throw error;
			}

			const errorType = this.classifyError(error);

			// Same error classification as non-streaming
			if (errorType === ErrorType.AUTHENTICATION || errorType === ErrorType.MODEL_NOT_FOUND) {
				throw error;
			}

			this.logger?.warn(
				`Primary streaming provider failed (${errorType}), attempting fallback...`,
				error instanceof Error ? error.message : String(error)
			);

			// Show fallback as a tool-execution style message in Agent UI if available
			try {
				const errMsg = error instanceof Error ? error.message : String(error);
				try {
					const modelName = (request as any)?.model || '';
					const message = `Falling back to ${this.fallbackName} : ${modelName || 'unknown model'}`;
					const execId = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
					if (
						this.plugin &&
						(this.plugin as any).agentView &&
						typeof (this.plugin as any).agentView.showToolExecution === 'function'
					) {
						// @ts-ignore
						await (this.plugin as any).agentView.showToolExecution(
							'provider-fallback',
							{
								primary: this.primaryName,
								fallback: this.fallbackName,
								model: modelName,
								message,
								error: errMsg,
							},
							execId
						);
					}
				} catch (e) {
					// ignore UI errors
				}
			} catch (e) {
				// ignore UI errors
			}

			// Only allow a single fallback attempt per request
			const alreadyFellBack = (request as any)?.__fallbackAttempted === true;
			if (alreadyFellBack) {
				throw error;
			}
			try {
				(request as any).__fallbackAttempted = true;
			} catch (e) {
				// ignore
			}

			try {
				const response = this.fallback.generateStreamingResponse?.(request, onChunk);
				if (!response) {
					// Fallback doesn't support streaming, fall back to non-streaming
					const resp = await this.fallback.generateModelResponse(request);
					try {
						const execId = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
						if (
							this.plugin &&
							(this.plugin as any).agentView &&
							typeof (this.plugin as any).agentView.showToolResult === 'function'
						) {
							// @ts-ignore
							await (this.plugin as any).agentView.showToolResult(
								'provider-fallback',
								{ success: true, data: { primary: this.primaryName, fallback: this.fallbackName } },
								execId
							);
						}
					} catch (_) {
						// ignore UI errors
					}
					return resp;
				}
				const resp = await response.complete;
				try {
					const execId = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
					if (
						this.plugin &&
						(this.plugin as any).agentView &&
						typeof (this.plugin as any).agentView.showToolResult === 'function'
					) {
						// @ts-ignore
						await (this.plugin as any).agentView.showToolResult(
							'provider-fallback',
							{ success: true, data: { primary: this.primaryName, fallback: this.fallbackName } },
							execId
						);
					}
				} catch (_) {
					// ignore UI errors
				}
				return resp;
			} catch (fallbackError) {
				this.logger?.error('Fallback streaming provider also failed', fallbackError);
				try {
					const execId = `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
					if (
						this.plugin &&
						(this.plugin as any).agentView &&
						typeof (this.plugin as any).agentView.showToolResult === 'function'
					) {
						// @ts-ignore
						await (this.plugin as any).agentView.showToolResult(
							'provider-fallback',
							{ success: false, error: String(fallbackError) },
							execId
						);
					}
				} catch (e) {
					// ignore
				}
				throw error;
			}
		}
	}

	/**
	 * Classify the error type to determine if fallback should be attempted
	 *
	 * @param error - The error to classify
	 * @returns ErrorType enum value
	 */
	private classifyError(error: unknown): ErrorType {
		const errorStr = error instanceof Error ? error.message : String(error);

		// Check for network errors (case-insensitive, check for natural language patterns)
		if (
			errorStr.toLowerCase().includes('network') ||
			errorStr.toLowerCase().includes('failed to fetch') ||
			errorStr.toLowerCase().includes('connection') ||
			errorStr.toLowerCase().includes('timeout')
		) {
			return ErrorType.NETWORK;
		}

		// Check for rate limiting
		if (
			errorStr.includes('429') ||
			errorStr.toLowerCase().includes('rate limit') ||
			errorStr.toLowerCase().includes('too many requests')
		) {
			return ErrorType.RATE_LIMIT;
		}

		// Check for authentication errors
		if (
			errorStr.includes('401') ||
			errorStr.toLowerCase().includes('unauthorized') ||
			errorStr.toLowerCase().includes('invalid api key') ||
			errorStr.toLowerCase().includes('authentication')
		) {
			return ErrorType.AUTHENTICATION;
		}

		// Check for model not found
		if (
			errorStr.includes('404') ||
			errorStr.toLowerCase().includes('model not found') ||
			errorStr.toLowerCase().includes('not found')
		) {
			return ErrorType.MODEL_NOT_FOUND;
		}

		// Check for API errors (5xx)
		if (
			errorStr.includes('500') ||
			errorStr.includes('502') ||
			errorStr.includes('503') ||
			errorStr.includes('504') ||
			errorStr.toLowerCase().includes('internal server error') ||
			errorStr.toLowerCase().includes('service unavailable')
		) {
			return ErrorType.API;
		}

		return ErrorType.OTHER;
	}
}
