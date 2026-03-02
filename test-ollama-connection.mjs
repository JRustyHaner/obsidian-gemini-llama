/**
 * Quick test script to verify Ollama connection
 */

const OLLAMA_ENDPOINT = 'http://100.71.158.16:11434';

async function testConnection() {
	console.log('Testing Ollama connection...');
	console.log(`Endpoint: ${OLLAMA_ENDPOINT}`);
	console.log('');

	// Test 1: Check if server is reachable
	try {
		console.log('1. Testing server health...');
		const healthResponse = await fetch(`${OLLAMA_ENDPOINT}/api/tags`);
		if (!healthResponse.ok) {
			throw new Error(`Server returned ${healthResponse.status}`);
		}
		const models = await healthResponse.json();
		console.log('✓ Server is reachable');
		console.log(`  Available models: ${models.models?.map((m) => m.name).join(', ') || 'none'}`);
		console.log('');
	} catch (error) {
		console.error('✗ Server health check failed:', error.message);
		return;
	}

	// Test 2: Try a simple generation request
	try {
		console.log('2. Testing generate endpoint...');
		const generateRequest = {
			model: 'llama2:7b',
			prompt: 'Say hello in one word:',
			stream: false,
		};

		const generateResponse = await fetch(`${OLLAMA_ENDPOINT}/api/generate`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(generateRequest),
		});

		if (!generateResponse.ok) {
			const errorText = await generateResponse.text();
			throw new Error(`Generate request failed: ${generateResponse.status} - ${errorText}`);
		}

		const result = await generateResponse.json();
		console.log('✓ Generate endpoint working');
		console.log(`  Response: ${result.response || 'no response'}`);
		console.log('');
	} catch (error) {
		console.error('✗ Generate test failed:', error.message);
		return;
	}

	// Test 3: Try embeddings endpoint
	try {
		console.log('3. Testing embeddings endpoint...');
		const embeddingRequest = {
			model: 'nomic-embed-text',
			prompt: 'test',
		};

		const embeddingResponse = await fetch(`${OLLAMA_ENDPOINT}/api/embeddings`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(embeddingRequest),
		});

		if (!embeddingResponse.ok) {
			console.log('⚠ Embeddings endpoint not available (model may not be installed)');
			console.log('  This is optional - install with: ollama pull nomic-embed-text');
		} else {
			const result = await embeddingResponse.json();
			console.log('✓ Embeddings endpoint working');
			console.log(`  Vector dimensions: ${result.embedding?.length || 0}`);
		}
		console.log('');
	} catch (error) {
		console.log('⚠ Embeddings test skipped:', error.message);
		console.log('');
	}

	console.log('✓ All tests passed! Ollama is ready to use.');
}

testConnection().catch(console.error);
