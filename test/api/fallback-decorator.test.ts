/**
 * Tests for FallbackDecorator
 */

import { FallbackDecorator } from '../../src/api/fallback-decorator';
import {
	ModelApi,
	BaseModelRequest,
	ModelResponse,
	StreamingModelResponse,
	StreamCallback,
	StreamChunk,
} from '../../src/api/interfaces/model-api';

describe('FallbackDecorator', () => {
	let mockPrimaryClient: jest.Mocked<ModelApi>;
	let mockFallbackClient: jest.Mocked<ModelApi>;
	let mockLogger: any;

	beforeEach(() => {
		mockPrimaryClient = {
			generateModelResponse: jest.fn(),
			generateStreamingResponse: jest.fn(),
		} as jest.Mocked<ModelApi>;

		mockFallbackClient = {
			generateModelResponse: jest.fn(),
			generateStreamingResponse: jest.fn(),
		} as jest.Mocked<ModelApi>;

		mockLogger = {
			log: jest.fn(),
			debug: jest.fn(),
			warn: jest.fn(),
			error: jest.fn(),
		};
	});

	describe('generateModelResponse', () => {
		it('should use primary client when successful', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const expectedResponse: ModelResponse = { markdown: 'response', rendered: 'response' };

			mockPrimaryClient.generateModelResponse.mockResolvedValue(expectedResponse);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			const result = await decorator.generateModelResponse(request);

			expect(result).toEqual(expectedResponse);
			expect(mockPrimaryClient.generateModelResponse).toHaveBeenCalledWith(request);
			expect(mockFallbackClient.generateModelResponse).not.toHaveBeenCalled();
		});

		it('should fallback to secondary client on network error', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const expectedResponse: ModelResponse = { markdown: 'fallback response', rendered: 'fallback response' };

			mockPrimaryClient.generateModelResponse.mockImplementation(() => {
				throw new Error('Provider network issue');
			});
			mockFallbackClient.generateModelResponse.mockResolvedValue(expectedResponse);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			const result = await decorator.generateModelResponse(request);

			expect(result).toEqual(expectedResponse);
			expect(mockPrimaryClient.generateModelResponse).toHaveBeenCalledWith(request);
			expect(mockFallbackClient.generateModelResponse).toHaveBeenCalledWith(request);
			expect(mockLogger.warn).toHaveBeenCalled();
		});

		it('should not fallback on authentication error', async () => {
			const request: BaseModelRequest = { prompt: 'test' };

			mockPrimaryClient.generateModelResponse.mockRejectedValue(new Error('Authentication failed: invalid api key'));

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			await expect(decorator.generateModelResponse(request)).rejects.toThrow('invalid api key');
			expect(mockFallbackClient.generateModelResponse).not.toHaveBeenCalled();
		});

		it('should not fallback on model not found error', async () => {
			const request: BaseModelRequest = { prompt: 'test' };

			mockPrimaryClient.generateModelResponse.mockRejectedValue(new Error('Model not found: model does not exist'));

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			await expect(decorator.generateModelResponse(request)).rejects.toThrow('model does not exist');
			expect(mockFallbackClient.generateModelResponse).not.toHaveBeenCalled();
		});

		it('should throw primary error if fallback also fails', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const primaryError = new Error('Primary network error');

			mockPrimaryClient.generateModelResponse.mockRejectedValue(primaryError);
			mockFallbackClient.generateModelResponse.mockRejectedValue(new Error('Fallback also failed'));

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			await expect(decorator.generateModelResponse(request)).rejects.toThrow('Primary network error');
			expect(mockLogger.error).toHaveBeenCalled();
		});

		it('should fallback on rate limit error', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const expectedResponse: ModelResponse = { markdown: 'response', rendered: 'response' };

			mockPrimaryClient.generateModelResponse.mockRejectedValue(new Error('Rate limit exceeded'));
			mockFallbackClient.generateModelResponse.mockResolvedValue(expectedResponse);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			const result = await decorator.generateModelResponse(request);

			expect(result).toEqual(expectedResponse);
			expect(mockFallbackClient.generateModelResponse).toHaveBeenCalled();
		});

		it('should fallback on API server error', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const expectedResponse: ModelResponse = { markdown: 'response', rendered: 'response' };

			mockPrimaryClient.generateModelResponse.mockRejectedValue(new Error('Internal server error'));
			mockFallbackClient.generateModelResponse.mockResolvedValue(expectedResponse);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			const result = await decorator.generateModelResponse(request);

			expect(result).toEqual(expectedResponse);
			expect(mockFallbackClient.generateModelResponse).toHaveBeenCalled();
		});

		it('should handle no fallback client gracefully', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const primaryError = new Error('Network error');

			mockPrimaryClient.generateModelResponse.mockRejectedValue(primaryError);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: null,
				logger: mockLogger,
			});

			await expect(decorator.generateModelResponse(request)).rejects.toThrow('Network error');
		});
	});

	describe('generateStreamingResponse', () => {
		it('should use primary client when successful', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const expectedResponse: ModelResponse = { markdown: 'chunk1chunk2', rendered: 'chunk1chunk2' };

			const mockStreamingResponse: StreamingModelResponse = {
				complete: Promise.resolve(expectedResponse),
				cancel: jest.fn(),
			};

			(mockPrimaryClient.generateStreamingResponse as jest.Mock).mockReturnValue(mockStreamingResponse);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			const onChunk = jest.fn();
			const result = decorator.generateStreamingResponse(request, onChunk);

			expect(result).toBeDefined();
			expect(result.cancel).toBeDefined();

			const response = await result.complete;
			expect(response).toEqual(expectedResponse);
			expect(mockFallbackClient.generateStreamingResponse).not.toHaveBeenCalled();
		});

		it('should fallback to secondary client on network error', async () => {
			const request: BaseModelRequest = { prompt: 'test' };
			const expectedResponse: ModelResponse = { markdown: 'fallback response', rendered: 'fallback response' };

			(mockPrimaryClient.generateStreamingResponse as jest.Mock).mockImplementation(() => {
				throw new Error('Provider network issue');
			});

			const mockFallbackStreamingResponse: StreamingModelResponse = {
				complete: Promise.resolve(expectedResponse),
				cancel: jest.fn(),
			};

			(mockFallbackClient.generateStreamingResponse as jest.Mock).mockReturnValue(mockFallbackStreamingResponse);

			const decorator = new FallbackDecorator({
				primary: mockPrimaryClient,
				fallback: mockFallbackClient,
				logger: mockLogger,
			});

			const onChunk = jest.fn();
			const result = decorator.generateStreamingResponse(request, onChunk);

			expect(result).toBeDefined();

			const response = await result.complete;
			expect(response).toEqual(expectedResponse);
			expect(mockLogger.warn).toHaveBeenCalled();
		});
	});
});
