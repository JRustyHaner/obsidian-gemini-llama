/**
 * API module for Gemini AI integration
 */

// Re-export the interfaces
export type {
	ModelApi,
	ModelResponse,
	BaseModelRequest,
	ExtendedModelRequest,
	InlineDataPart,
	ImagePart,
	ToolCall,
	ToolDefinition,
} from './interfaces/model-api';

// Provider types
export { ModelProvider, RagProvider } from './types';
export type { ModelCapabilities, ProviderModelConfig } from './types';

// Export the simplified factory
export { GeminiClientFactory, ModelUseCase } from './simple-factory';

// Export the Gemini client
export { GeminiClient } from './gemini-client';
export type { GeminiClientConfig } from './gemini-client';

// Export the Ollama client
export { OllamaClient } from './ollama-client';
export type { OllamaClientConfig, OllamaEmbeddingRequest, OllamaEmbeddingResponse } from './ollama-client';

// Export decorators
export { RetryDecorator } from './retry-decorator';
export { FallbackDecorator } from './fallback-decorator';
export type { FallbackConfig } from './fallback-decorator';
