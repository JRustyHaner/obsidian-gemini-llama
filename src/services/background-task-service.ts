import { Plugin } from 'obsidian';
import { Logger } from '../utils/logger';
import { Notice } from 'obsidian';
import type ObsidianGemini from '../main';
import { GeminiClientFactory } from '../api';
import { ModelUseCase } from '../api/simple-factory';

export interface BackgroundTask {
	id: string;
	type: 'chat' | 'summary' | 'rewrite' | 'custom';
	prompt: string;
	context?: string;
	filePath?: string;
	createdAt: Date;
	status: 'queued' | 'processing' | 'completed' | 'failed';
	result?: string;
	error?: string;
}

export class BackgroundTaskService {
	private plugin: ObsidianGemini;
	private logger: Logger;
	private tasks: Map<string, BackgroundTask> = new Map();
	private isProcessing: boolean = false;
	private processingQueue: BackgroundTask[] = [];

	constructor(plugin: ObsidianGemini, logger: Logger) {
		this.plugin = plugin;
		this.logger = logger;
	}

	/**
	 * Add a task to the background queue
	 */
	async addTask(task: Omit<BackgroundTask, 'id' | 'createdAt' | 'status'>): Promise<string> {
		const taskId = `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
		const fullTask: BackgroundTask = {
			...task,
			id: taskId,
			createdAt: new Date(),
			status: 'queued',
		};

		this.tasks.set(taskId, fullTask);
		this.processingQueue.push(fullTask);

		this.logger.log(`Added background task: ${taskId} (${task.type})`);

		// Start processing if not already running
		if (!this.isProcessing) {
			this.startProcessing();
		}

		return taskId;
	}

	/**
	 * Get a task by ID
	 */
	getTask(taskId: string): BackgroundTask | undefined {
		return this.tasks.get(taskId);
	}

	/**
	 * Get all tasks
	 */
	getAllTasks(): BackgroundTask[] {
		return Array.from(this.tasks.values());
	}

	/**
	 * Start processing the queue
	 */
	private async startProcessing(): Promise<void> {
		if (this.isProcessing) return;

		this.isProcessing = true;
		this.logger.log('Started background task processing');

		while (this.processingQueue.length > 0) {
			const task = this.processingQueue.shift();
			if (!task) continue;

			try {
				task.status = 'processing';
				this.logger.log(`Processing task: ${task.id}`);

				// Process the task based on type
				const result = await this.processTask(task);

				task.status = 'completed';
				task.result = result;

				this.logger.log(`Completed task: ${task.id}`);

				// Send notification
				this.sendNotification(task);
			} catch (error) {
				task.status = 'failed';
				task.error = error instanceof Error ? error.message : 'Unknown error';
				this.logger.error(`Failed task: ${task.id}`, error);
			}
		}

		this.isProcessing = false;
		this.logger.log('Stopped background task processing');
	}

	/**
	 * Process a single task
	 */
	private async processTask(task: BackgroundTask): Promise<string> {
		// Import the necessary services dynamically to avoid circular dependencies
		const { GeminiClientFactory } = await import('../api');

		switch (task.type) {
			case 'chat': {
				const client = GeminiClientFactory.createFromPlugin(this.plugin, ModelUseCase.CHAT);
				const response = await client.generateModelResponse({ prompt: task.prompt });
				return response.markdown;
			}

			case 'summary': {
				const client = GeminiClientFactory.createFromPlugin(this.plugin, ModelUseCase.SUMMARY);
				const prompt = `Please summarize the following content:\n\n${task.context || task.prompt}`;
				const response = await client.generateModelResponse({ prompt });
				return response.markdown;
			}

			case 'rewrite': {
				const client = GeminiClientFactory.createFromPlugin(this.plugin, ModelUseCase.REWRITE);
				const prompt = `Please rewrite the following text:\n\n${task.context || task.prompt}`;
				const response = await client.generateModelResponse({ prompt });
				return response.markdown;
			}

			case 'custom': {
				const client = GeminiClientFactory.createFromPlugin(this.plugin, ModelUseCase.CHAT);
				const response = await client.generateModelResponse({ prompt: task.prompt });
				return response.markdown;
			}

			default:
				throw new Error(`Unknown task type: ${task.type}`);
		}
	}

	/**
	 * Send a system notification when task completes
	 */
	private sendNotification(task: BackgroundTask): void {
		// Check if notifications are enabled
		if (!(this.plugin as any).settings?.backgroundNotificationsEnabled) {
			// Still show in-app notice
			new Notice(`Background task completed: ${task.type}`, 5000);
			return;
		}

		try {
			// Check if we're on mobile - Obsidian has different notification handling
			const isMobile = (this.plugin.app as any).isMobile;

			if (isMobile) {
				// On mobile, just use in-app notice with more prominent styling
				new Notice(`Background task completed: ${task.type}`, 10000); // Show for 10 seconds
				return;
			}

			// Desktop: Use system notifications
			const notifier = require('node-notifier');

			const title = `Gemini Scribe: ${task.type.charAt(0).toUpperCase() + task.type.slice(1)} Complete`;
			let message = `Your ${task.type} task has finished.`;

			if (task.result && task.result.length > 100) {
				message += ` Result: ${task.result.substring(0, 100)}...`;
			} else if (task.result) {
				message += ` Result: ${task.result}`;
			}

			notifier.notify({
				title,
				message,
				sound: true, // Play notification sound
				wait: false, // Don't wait for user action
			});

			// Also show in-app notice
			new Notice(`Background task completed: ${task.type}`, 5000);
		} catch (error) {
			this.logger.error('Failed to send notification:', error);
			// Fallback to in-app notice only
			new Notice(`Background task completed: ${task.type}`, 5000);
		}
	}

	/**
	 * Clear completed/failed tasks older than specified hours
	 */
	clearOldTasks(hours: number = 24): void {
		const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
		const toDelete: string[] = [];

		for (const [id, task] of this.tasks) {
			if ((task.status === 'completed' || task.status === 'failed') && task.createdAt < cutoff) {
				toDelete.push(id);
			}
		}

		toDelete.forEach((id) => this.tasks.delete(id));
		this.logger.log(`Cleared ${toDelete.length} old tasks`);
	}

	/**
	 * Stop processing and clear queue
	 */
	stop(): void {
		this.isProcessing = false;
		this.processingQueue = [];
		this.logger.log('Background task service stopped');
	}
}
