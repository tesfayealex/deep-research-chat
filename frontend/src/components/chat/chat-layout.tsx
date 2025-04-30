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
  plan?: string[];
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
        plan: [],
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
                            updatedMsg.plan = update.data;
                            updatedMsg.status = 'Executing plan...'; // Update status
                            updatedMsg.text = updatedMsg.text || '';
                        } else if (update.type === 'step_result') {
                            // Store step results in the thinking array
                            updatedMsg.thinking = updatedMsg.thinking || [];
                            updatedMsg.thinking.push(update.data);
                            updatedMsg.status = 'Processing results...'; // Update status
                            updatedMsg.text = updatedMsg.text || '';
                        } else if (update.type === 'report') {
                            updatedMsg.text = update.data; // Final report content
                            updatedMsg.status = 'Finished';
                            updatedMsg.isStreaming = false; // Mark as finished
                            // Don't clear thinking data here
                        } else if (update.type === 'error') {
                            updatedMsg.text = `Error: ${update.data}`;
                            updatedMsg.status = 'Error';
                            updatedMsg.isStreaming = false;
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