'use client'

import React, { useState, useEffect, useRef } from 'react';
import ChatMessages from './chat-messages';
import ChatInput from './chat-input';
import { v4 as uuidv4 } from 'uuid'; // For generating unique message IDs

// Define message structure including streaming state
export interface Message {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  // Optional fields for streaming progress
  isStreaming?: boolean;
  plan?: { step_name: string; step_detail: string }[]; 
  status?: string;
  thinking?: any[]; // Add thinking field to store step results
}

export default function ChatLayout() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Use a ref to manage updates to the streaming message
  const streamingMessageRef = useRef<Message | null>(null);

  const handleSendMessage = async (inputText: string) => {
    const userMessage: Message = {
      id: uuidv4(),
      sender: 'user',
      text: inputText,
    };
    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Create a placeholder for the agent's response
    const agentMessageId = uuidv4();
    const initialAgentMessage: Message = {
        id: agentMessageId,
        sender: 'agent',
        text: '', // Start empty
        isStreaming: true,
        status: 'Connecting...',
        plan: [], // Initialize as empty array
        thinking: [], // Initialize thinking
    };
    // Add placeholder agent message
    setMessages(prev => [...prev, initialAgentMessage]);
    streamingMessageRef.current = initialAgentMessage;

    // --- Real API Call (Streaming) --- 
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${backendUrl}/api/research`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputText }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      if (!response.body) {
        throw new Error("Response body is null");
      }

      // Process the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        console.log("Stream reader.read():", { done }); // Log if stream ends
        if (done) {
            console.log("Stream finished.");
            break;
        }

        const decodedChunk = decoder.decode(value, { stream: true });
        console.log("Stream decodedChunk:", decodedChunk); // Log raw decoded chunk
        buffer += decodedChunk;

        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
            if (line.trim() === '') continue;
            console.log("Processing line:", line); // Log line before parsing
            try {
                const update = JSON.parse(line);
                console.log('Received stream update:', update);

                // Update the streaming message state
                setMessages(prev => prev.map(msg => {
                    if (msg.id === agentMessageId) {
                        const updatedMsg = { ...msg };
                        if (update.type === 'status') {
                            updatedMsg.status = update.data;
                            updatedMsg.text = updatedMsg.text || ''; // Ensure text isn't null
                        } else if (update.type === 'plan') {
                            // Store the structured plan
                            updatedMsg.plan = update.data;
                            // Format the plan for display in the text field for now
                            // We will refine this in message-bubble.tsx later
                            const planText = update.data.map((step: { step_name: string, step_detail: string }, index: number) => 
                                `${index + 1}. ${step.step_name}: ${step.step_detail}`
                            ).join('\n');
                            updatedMsg.text = `Okay, here is the plan:\n${planText}`;
                            updatedMsg.status = 'Executing plan...'; // Update status
                        } else if (update.type === 'step_result') {
                            // Store step results in the thinking array
                            updatedMsg.thinking = updatedMsg.thinking || [];
                            // Make sure we don't have duplicates if stream resends
                            if (!updatedMsg.thinking.some(t => t.step_index === update.data.step_index)) {
                                updatedMsg.thinking.push(update.data);
                            }
                            // Display findings preview in the main text for now?
                            // updatedMsg.text += `\nStep ${update.data.step_index + 1} findings: ${update.data.findings_preview}`;
                            updatedMsg.status = `Executing Step ${update.data.step_index + 2}...`; // Update status
                            updatedMsg.text = updatedMsg.text || ''; // Ensure text isn't null
                        } else if (update.type === 'evaluation') {
                            updatedMsg.status = update.data.status;
                            // Optionally add evaluation text
                            // updatedMsg.text += `\nEvaluation: ${update.data.status}`;
                            updatedMsg.text = updatedMsg.text || '';
                        } else if (update.type === 'report') {
                            updatedMsg.text = update.data; // Final report content
                            updatedMsg.status = 'Finished';
                            updatedMsg.isStreaming = false; // Mark as finished
                            updatedMsg.thinking = updatedMsg.thinking || []; // Ensure thinking is array
                        } else if (update.type === 'error') {
                            updatedMsg.text = `Error: ${update.data}`;
                            updatedMsg.status = 'Error';
                            updatedMsg.isStreaming = false;
                        } else if (update.type === 'complete') {
                            // Stream ended normally, mark streaming finished if not already
                            updatedMsg.isStreaming = false;
                            if (!updatedMsg.status || updatedMsg.status.includes('...')) {
                                updatedMsg.status = 'Finished'; // Set final status if needed
                            } 
                        }
                        // Update the ref as well for potential direct access if needed
                        if (streamingMessageRef.current?.id === agentMessageId) {
                            streamingMessageRef.current = updatedMsg;
                        }
                        return updatedMsg;
                    }
                    return msg;
                }));
            } catch (e) {
                console.error("Error parsing stream line:", line, e);
            }
        }
      }

    } catch (error) {
        console.error("Error fetching or processing stream:", error);
        // Update the placeholder message to show the error
        setMessages(prev => prev.map(msg => {
            if (msg.id === agentMessageId) {
                return {
                    ...msg,
                    text: 'Sorry, I encountered an error connecting to the backend or processing the stream.',
                    status: 'Error',
                    isStreaming: false,
                };
            }
            return msg;
        }));
        streamingMessageRef.current = null;
    } finally {
        setIsLoading(false);
        // Ensure streaming is marked false if not already done by 'report' or 'error' type
        setMessages(prev => prev.map(msg => (
            msg.id === agentMessageId && msg.isStreaming ? 
                { ...msg, isStreaming: false, status: msg.status === 'Connecting...' || msg.status === 'Generating plan...' ? 'Interrupted' : msg.status } 
                : msg
        )));
        // Only clear the streaming message ref, not the message data itself
        if (streamingMessageRef.current?.id === agentMessageId) {
            streamingMessageRef.current = null;
        }
    }
  };

  // Placeholder effect for potential future use (e.g., loading initial messages)
  useEffect(() => {
    // console.log("ChatLayout mounted");
  }, []);

  return (
    <div className="flex flex-col h-screen max-h-screen bg-background">
      {/* Header (Optional - can add later) */}
      {/* <header className="p-4 border-b">Header</header> */}

      {/* Message Area */}
      <ChatMessages messages={messages} />

      {/* Input Area */}
      <div className="p-4 border-t bg-background">
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
} 