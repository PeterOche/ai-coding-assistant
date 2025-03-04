// The module 'vscode' contains the VS Code extensibility API
import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';

// This method is called when your extension is activated
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "ai-coding-assistant" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	const helloWorldDisposable = vscode.commands.registerCommand('ai-coding-assistant.helloWorld', () => {
		// The code you place here will be executed every time your command is executed
		// Display a message box to the user
		vscode.window.showInformationMessage('Hello World from ai-coding-assistant!');
	});

	// Register the askAI command
	const askAIDisposable = vscode.commands.registerCommand('ai-coding-assistant.askAI', async () => {
		// Prompt the user for input
		const userInput = await vscode.window.showInputBox({
			prompt: 'What would you like to ask the AI?',
			placeHolder: 'e.g., Write a Python function to reverse a string'
		});

		// Check if the user provided input or canceled the dialog
		if (!userInput) {
			return; // User canceled the input
		}

		// Show a progress notification
		vscode.window.withProgress({
			location: vscode.ProgressLocation.Notification,
			title: "AI is thinking...",
			cancellable: false
		}, async (progress) => {
			try {
				// Get the path to the Python script
				const scriptPath = path.join(context.extensionPath, 'src', 'deepseek_local.py');
				
				// Execute the Python script with the user's input
				const pythonProcess = child_process.spawn('python', [scriptPath, userInput]);
				
				let stdoutData = '';
				let stderrData = '';
				
				// Collect stdout data
				pythonProcess.stdout.on('data', (data) => {
					stdoutData += data.toString();
				});
				
				// Collect stderr data
				pythonProcess.stderr.on('data', (data) => {
					stderrData += data.toString();
				});
				
				// Handle process completion
				return new Promise<void>((resolve) => {
					pythonProcess.on('close', (code) => {
						if (code === 0) {
							// Extract the generated code from the output
							const outputLines = stdoutData.split('\n');
							let generatedCode = '';
							let inCodeSection = false;
							
							for (const line of outputLines) {
								if (line.includes('GENERATED CODE:')) {
									inCodeSection = true;
								} else if (inCodeSection && line.trim() !== '==================================================') {
									generatedCode += line + '\n';
								}
							}
							
							// Show the result in a new editor
							if (generatedCode.trim()) {
								const document = vscode.workspace.openTextDocument({
									content: generatedCode.trim(),
									language: 'python' // Assuming Python output, adjust as needed
								});
								document.then(doc => {
									vscode.window.showTextDocument(doc);
								});
								
								vscode.window.showInformationMessage('AI response generated successfully!');
							} else {
								vscode.window.showInformationMessage('AI response: ' + stdoutData);
							}
						} else {
							vscode.window.showErrorMessage(`Error running AI model: ${stderrData}`);
						}
						resolve();
					});
				});
			} catch (error) {
				vscode.window.showErrorMessage(`Failed to run AI model: ${error instanceof Error ? error.message : String(error)}`);
			}
		});
	});

	context.subscriptions.push(helloWorldDisposable);
	context.subscriptions.push(askAIDisposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}