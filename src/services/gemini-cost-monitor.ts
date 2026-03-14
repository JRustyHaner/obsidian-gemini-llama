export interface GeminiCostTotals {
	requests: number;
	tokens: number;
	estimatedCostUsd: number;
	lastRequest?: number;
}

export class GeminiCostMonitor {
	private totals: GeminiCostTotals;

	constructor() {
		this.totals = {
			requests: 0,
			tokens: 0,
			estimatedCostUsd: 0,
		};
	}

	recordRequest(tokens: number, estimatedCostUsd: number) {
		this.totals.requests += 1;
		this.totals.tokens += tokens;
		this.totals.estimatedCostUsd += estimatedCostUsd;
		this.totals.lastRequest = Date.now();
	}

	getTotals(): GeminiCostTotals {
		return { ...this.totals };
	}

	reset() {
		this.totals = {
			requests: 0,
			tokens: 0,
			estimatedCostUsd: 0,
		};
	}
}
