import { App, Modal, Setting, Notice } from 'obsidian';
import { BackgroundTaskService, BackgroundTask } from '../services/background-task-service';

export class BackgroundTasksModal extends Modal {
	private taskService: BackgroundTaskService;
	private refreshInterval: NodeJS.Timeout | null = null;

	constructor(app: App, taskService: BackgroundTaskService) {
		super(app);
		this.taskService = taskService;
	}

	onOpen() {
		const { contentEl } = this;
		contentEl.empty();

		contentEl.createEl('h2', { text: 'Background Tasks' });

		// Create container for task list
		const taskListContainer = contentEl.createDiv({ cls: 'background-tasks-list' });

		// Refresh button
		new Setting(contentEl)
			.addButton((button) =>
				button
					.setButtonText('Refresh')
					.setCta()
					.onClick(() => this.refreshTaskList(taskListContainer))
			)
			.addButton((button) =>
				button
					.setButtonText('Clear Completed')
					.setWarning()
					.onClick(() => {
						this.taskService.clearOldTasks(0); // Clear all completed/failed
						this.refreshTaskList(taskListContainer);
						new Notice('Cleared completed and failed tasks');
					})
			);

		// Initial refresh
		this.refreshTaskList(taskListContainer);

		// Auto-refresh every 5 seconds
		this.refreshInterval = setInterval(() => {
			this.refreshTaskList(taskListContainer);
		}, 5000);
	}

	onClose() {
		const { contentEl } = this;
		contentEl.empty();

		// Clear auto-refresh
		if (this.refreshInterval) {
			clearInterval(this.refreshInterval);
			this.refreshInterval = null;
		}
	}

	private refreshTaskList(container: Element) {
		container.empty();

		const tasks = this.taskService.getAllTasks();

		if (tasks.length === 0) {
			container.createEl('p', {
				text: 'No background tasks yet. Use the commands to queue tasks.',
				cls: 'background-tasks-empty',
			});
			return;
		}

		// Sort tasks by creation date (newest first)
		tasks.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

		for (const task of tasks) {
			this.createTaskItem(container, task);
		}
	}

	private createTaskItem(container: Element, task: BackgroundTask) {
		const taskItem = container.createDiv({ cls: 'background-task-item' });

		// Task header
		const header = taskItem.createDiv({ cls: 'background-task-header' });
		header.createSpan({
			text: `${task.type.charAt(0).toUpperCase() + task.type.slice(1)}`,
			cls: 'background-task-type',
		});
		header.createSpan({ text: this.formatDate(task.createdAt), cls: 'background-task-date' });

		// Status badge
		const statusBadge = header.createSpan({ cls: `background-task-status status-${task.status}` });
		statusBadge.setText(task.status);

		// Task content preview
		const content = taskItem.createDiv({ cls: 'background-task-content' });
		const promptPreview = task.prompt.length > 100 ? task.prompt.substring(0, 100) + '...' : task.prompt;
		content.createEl('p', { text: promptPreview, cls: 'background-task-prompt' });

		if (task.context) {
			const contextPreview = task.context.length > 100 ? task.context.substring(0, 100) + '...' : task.context;
			content.createEl('p', { text: `Context: ${contextPreview}`, cls: 'background-task-context' });
		}

		// Result or error
		if (task.result) {
			const resultDiv = taskItem.createDiv({ cls: 'background-task-result' });
			resultDiv.createEl('h4', { text: 'Result:' });
			const resultPreview = task.result.length > 200 ? task.result.substring(0, 200) + '...' : task.result;
			resultDiv.createEl('p', { text: resultPreview });
		}

		if (task.error) {
			const errorDiv = taskItem.createDiv({ cls: 'background-task-error' });
			errorDiv.createEl('h4', { text: 'Error:' });
			errorDiv.createEl('p', { text: task.error });
		}

		// Actions
		const actions = taskItem.createDiv({ cls: 'background-task-actions' });

		if (task.status === 'completed' && task.result) {
			actions
				.createEl('button', {
					text: 'Copy Result',
					cls: 'background-task-action-btn',
					type: 'button',
				})
				.addEventListener('click', () => {
					navigator.clipboard.writeText(task.result!);
					new Notice('Result copied to clipboard');
				});
		}

		if (task.filePath) {
			actions
				.createEl('button', {
					text: 'Open File',
					cls: 'background-task-action-btn',
					type: 'button',
				})
				.addEventListener('click', () => {
					const file = this.app.vault.getAbstractFileByPath(task.filePath!);
					if (file) {
						this.app.workspace.openLinkText('', task.filePath!);
					} else {
						new Notice('File not found');
					}
				});
		}
	}

	private formatDate(date: Date): string {
		const now = new Date();
		const diffMs = now.getTime() - date.getTime();
		const diffMins = Math.floor(diffMs / (1000 * 60));
		const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
		const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

		if (diffMins < 1) return 'Just now';
		if (diffMins < 60) return `${diffMins}m ago`;
		if (diffHours < 24) return `${diffHours}h ago`;
		if (diffDays < 7) return `${diffDays}d ago`;

		return date.toLocaleDateString();
	}
}
