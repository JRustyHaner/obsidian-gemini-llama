/**
 * Provider types for the plugin
 */

/**
 * Available AI model providers
 */
export enum ModelProvider {
	GEMINI = 'gemini',
	OLLAMA = 'ollama',
}

/**
 * Available RAG (Retrieval Augmented Generation) providers
 */
export enum RagProvider {
	GEMINI = 'gemini', // Google File Search API
	OLLAMA = 'ollama', // Local embeddings with Ollama
}

/**
 * Model capabilities for feature detection
 */
export interface ModelCapabilities {
	supportsVision: boolean;
	supportsGrounding: boolean;
	supportsToolCalling: boolean;
	supportsCloudRag: boolean;
	supportsLocalRag: boolean;
}

/**
 * Provider-specific model configuration
 */
export interface ProviderModelConfig {
	chat?: string;
	summary?: string;
	completions?: string;
	rewrite?: string;
	embedding?: string;
}
