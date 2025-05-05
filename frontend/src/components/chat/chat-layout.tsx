'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react';
import ChatMessages from './chat-messages';
import ChatInput from './chat-input';
import { v4 as uuidv4 } from 'uuid'; // For generating unique message IDs
import { Sidebar } from './sidebar';

// Define StepResult type (matches backend step_result data)
export interface StepResult {
  step_index: number;
  step_name?: string;
  findings_preview?: string;
  sources?: string[];
  [key: string]: any; 
}

// Define Message structure (used by UI components)
export interface Message {
  id: string;
  sender: 'user' | 'agent';
  text?: string; // Optional: Used for user messages, final report, status, error
  isStreaming?: boolean;
  plan?: { step_name: string; step_detail: string }[]; // Used by plan yields
  status?: string; // Used by status/evaluation yields
  thinking?: StepResult[]; // Accumulates step_result yields
  // Add field to store original event type for debugging/styling
  originalEventType?: string; 
  originalEventName?: string;
}

// Define History Event structure matching backend response from /api/history/{id}
interface HistoryEvent {
  conversation_id: string;
  timestamp: string; 
  type: 'user_message' | 'agent_yield'; // The main types we fetch
  sender?: string;
  name?: string; // Stores the original yield type (plan, step_result, etc.) for agent_yield
  tags?: string[];
  data_json: string; // The JSON string data payload
}

// Define props for ChatLayout
interface ChatLayoutProps {
  selectedConversationId: string | null;
  onNewChat: () => void; // Function to clear selection
}

export function ChatLayout({ selectedConversationId, onNewChat }: ChatLayoutProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const streamingMessageRef = useRef<Message | null>(null);
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000';

  // Function to format history events into Message objects for UI
  const formatHistoryEvent = (event: HistoryEvent): Message | null => {
    try {
      const data = JSON.parse(event.data_json);
      const baseMessage: Partial<Message> = {
        id: `${event.conversation_id}-${event.timestamp}-${event.name || event.type}`,
        sender: event.sender === 'user' ? 'user' : 'agent',
        originalEventType: event.type,
        originalEventName: event.name, // Store the yield name
      };

      if (event.type === 'user_message') {
        return { 
          ...baseMessage, 
          sender: 'user',
          text: data.content || 'User message error' 
        } as Message;
      } 
      
      if (event.type === 'agent_yield') {
        // Use event.name (the original yield type) to map data_json content
        const yieldName = event.name;
        switch (yieldName) {
          case 'plan':
            return { ...baseMessage, plan: data, status: 'Plan Received' } as Message;
          case 'step_result':
            // For history, store the step result in the 'thinking' array
            // Create a message per step for simplicity in history view
            return { 
                ...baseMessage, 
                thinking: [data], // Store the single step result here
                // Optionally add a minimal status/text if needed
                status: `Step ${data.step_index + 1} Result`,
                // text: `Results for step: ${data.step_name || data.step_index + 1}`,
            } as Message;
          case 'report':
            // Final report is just text
            return { ...baseMessage, text: data, status: 'Report Generated' } as Message; 
          case 'status':
             // Display status messages clearly in history
             return { ...baseMessage, status: data, text: `Status: ${data}` } as Message;
          case 'evaluation':
             // Display evaluation status clearly
             return { ...baseMessage, status: data.status, text: `Evaluation: ${data.status}` } as Message;
          case 'error':
             // Display agent errors clearly
             return { ...baseMessage, text: `Agent Error: ${data}`, status: 'Error' } as Message;
          case 'complete':
             // Optionally show a completion message in history
             return { ...baseMessage, text: 'Agent processing finished.', status: 'Complete' } as Message;
             // return null; // Or hide completion markers
          default:
            // Handle unexpected or older yield names gracefully
            console.warn("Unhandled agent_yield name in history:", yieldName, data);
            return { 
                ...baseMessage, 
                text: `Agent action (${yieldName || 'Unknown'}): ${event.data_json}`,
                status: 'Unknown Agent Action' 
            } as Message;
        }
      }

      // Fallback for unexpected event types (shouldn't happen)
      console.warn("Unhandled history event type:", event.type);
      return null;

    } catch (e) {
      console.error("Error parsing history event data:", event.data_json, e);
      // Create an error message placeholder in the UI
      return {
        id: `${event.conversation_id}-${event.timestamp}-error`,
        sender: 'agent',
        text: 'Error displaying this part of the conversation.',
        originalEventType: event.type,
        status: 'Display Error'
      } as Message;
    }
  };

  // Function to load history (modified to handle multi-turn conversations)
  const loadHistory = useCallback(async (convId: string) => {
    console.log("Loading history for:", convId);
    setIsLoading(true);
    setMessages([]); // Clear existing messages
    try {
      const response = await fetch(`${backendUrl}/api/history/${convId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch history: ${response.status}`);
      }
      const historyEvents: HistoryEvent[] = await response.json();
      
      // Group events by conversation turn
      const userMessages: Message[] = [];
      const agentResponses: Message[] = [];
      const stepResultsByTurn: Map<number, StepResult[]> = new Map(); 
      const plansByTurn: Map<number, any> = new Map();
      
      let currentTurn = -1; // Increments with each user message
      
      // First pass: Process user messages to establish conversation turns
      historyEvents.forEach((event) => {
        if (event.type === 'user_message') {
          currentTurn++;
          try {
            const data = JSON.parse(event.data_json);
            userMessages.push({
              id: `${event.conversation_id}-${event.timestamp}-user-${currentTurn}`,
              sender: 'user',
              text: data.content || 'User message error',
              originalEventType: event.type,
            });
            // Initialize storage for this turn
            stepResultsByTurn.set(currentTurn, []);
          } catch (e) {
            console.error("Error parsing user message:", e);
          }
        }
      });
      
      // Second pass: Process agent yields in order
      currentTurn = -1; // Reset for second pass
      historyEvents.forEach((event) => {
        if (event.type === 'user_message') {
          currentTurn++; // Each user message starts a new turn
          return;
        }
        
        if (event.type === 'agent_yield' && currentTurn >= 0) {
          const yieldName = event.name;
          
          try {
            const data = JSON.parse(event.data_json);
            
            switch (yieldName) {
              case 'step_result':
                // Add to step results for current turn
                const stepResults = stepResultsByTurn.get(currentTurn) || [];
                stepResults.push(data);
                stepResultsByTurn.set(currentTurn, stepResults);
                break;
                
              case 'plan':
                // Store plan for current turn
                plansByTurn.set(currentTurn, data);
                break;
                
              case 'report':
                // Create agent response for the turn
                const agentResponse: Message = {
                  id: `${event.conversation_id}-${event.timestamp}-agent-${currentTurn}`,
                  sender: 'agent',
                  text: data, // Report content is the main text
                  originalEventType: event.type,
                  originalEventName: yieldName,
                };
                
                // Add plan if exists for this turn
                const plan = plansByTurn.get(currentTurn);
                if (plan) {
                  agentResponse.plan = plan;
                }
                
                // Add thinking (step results) if any
                const turnStepResults = stepResultsByTurn.get(currentTurn) || [];
                if (turnStepResults.length > 0) {
                  agentResponse.thinking = turnStepResults;
                }
                
                agentResponses.push(agentResponse);
                break;
                
              // Skip status, evaluation, error, complete events in history view
              // These are only relevant for live experience
            }
          } catch (e) {
            console.error(`Error processing ${yieldName} event:`, e);
          }
        }
      });
      
      // Combine into final ordered message list
      const formattedMessages: Message[] = [];
      // Interleave user messages and agent responses
      for (let turn = 0; turn < Math.max(userMessages.length, agentResponses.length); turn++) {
        if (turn < userMessages.length) {
          formattedMessages.push(userMessages[turn]);
        }
        if (turn < agentResponses.length) {
          formattedMessages.push(agentResponses[turn]);
        }
      }
      
      setMessages(formattedMessages);
      
    } catch (error) {
      console.error("Error loading history:", error);
      setMessages([{ id: uuidv4(), sender: 'agent', text: 'Error loading conversation history.'}]);
    } finally {
      setIsLoading(false);
    }
  }, [backendUrl]); // Dependencies

  // Effect to load history (remains the same)
  useEffect(() => {
    if (selectedConversationId) {
      loadHistory(selectedConversationId);
    } else {
      if (messages.length > 0) { // Check if messages exist before clearing
         setMessages([]);
      }
    }
  }, [selectedConversationId, loadHistory]); 

  // handleSendMessage (modified for streaming updates)
  const handleSendMessage = async (inputText: string) => {
    let currentConversationId = selectedConversationId;
    // If a history chat is loaded, start a new chat first
    if (currentConversationId) {
      onNewChat(); // Call the function to clear selection in parent
      // Reset local state immediately for UI responsiveness
      setMessages([]);
      currentConversationId = null; // Ensure we don't use the old ID
      // Wait briefly for state updates if necessary
      // await new Promise(resolve => setTimeout(resolve, 50)); 
    }

    const userMessage: Message = {
      id: uuidv4(),
      sender: 'user',
      text: inputText,
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Create a placeholder for the agent's response
    const agentMessageId = uuidv4();
    const initialAgentMessage: Message = {
        id: agentMessageId,
        sender: 'agent',
        text: '', 
        isStreaming: true,
        status: 'Connecting...',
        plan: [], 
        thinking: [],
    };
    setMessages(prev => [...prev, initialAgentMessage]);
    streamingMessageRef.current = initialAgentMessage;

    // --- Real API Call (Streaming) --- 
    try {
      const response = await fetch(`${backendUrl}/api/research`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputText }), // Send query
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
        if (done) {
            console.log("Stream finished.");
            break;
        }

        const decodedChunk = decoder.decode(value, { stream: true });
        buffer += decodedChunk;

        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
            if (line.trim() === '') continue;
            try {
                const update = JSON.parse(line);
                // console.log('Received stream update:', update);

                // Update the streaming message state based on update.type
                setMessages(prev => prev.map(msg => {
                    if (msg.id === agentMessageId) {
                        const updatedMsg = { ...msg };
                        switch (update.type) {
                            case 'status':
                                updatedMsg.status = update.data;
                                break;
                            case 'plan':
                                updatedMsg.plan = update.data;
                                updatedMsg.status = 'Executing plan...';
                                break;
                            case 'step_result':
                                updatedMsg.thinking = updatedMsg.thinking || [];
                                // Avoid duplicates if stream sends same step multiple times
                                if (!updatedMsg.thinking.some(t => t.step_index === update.data.step_index)) {
                                    updatedMsg.thinking.push(update.data);
                                }
                                updatedMsg.status = `Executing Step ${update.data.step_index + 2}...`;
                                break;
                            case 'evaluation':
                                updatedMsg.status = update.data.status;
                                break;
                            case 'report':
                                updatedMsg.text = update.data; // Final report content
                                updatedMsg.status = 'Finished';
                                updatedMsg.isStreaming = false;
                                updatedMsg.thinking = updatedMsg.thinking || [];
                                break;
                            case 'error':
                                updatedMsg.text = `Error: ${update.data}`;
                                updatedMsg.status = 'Error';
                                updatedMsg.isStreaming = false;
                                break;
                            case 'complete':
                                updatedMsg.isStreaming = false;
                                if (!updatedMsg.status || updatedMsg.status.includes('...')) {
                                    updatedMsg.status = 'Finished';
                                } 
                                break;
                            // Ignore heartbeat
                            case 'heartbeat': break;
                            default:
                                console.warn("Unhandled stream update type:", update.type);
                        }
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
        setMessages(prev => prev.map(msg => {
            if (msg.id === agentMessageId) {
                return { ...msg, text: 'Sorry, encountered an error.', status: 'Error', isStreaming: false };
            }
            return msg;
        }));
        streamingMessageRef.current = null;
    } finally {
        setIsLoading(false);
        setMessages(prev => prev.map(msg => (
            msg.id === agentMessageId && msg.isStreaming ? 
                { ...msg, isStreaming: false, status: msg.status?.includes('...') ? 'Interrupted' : msg.status } 
                : msg
        )));
        if (streamingMessageRef.current?.id === agentMessageId) {
            streamingMessageRef.current = null;
        }
    }
  };
  
  // handleSelectConversation (remains the same)
  const handleSelectConversation = (conversationId: string) => {
    if (selectedConversationId === conversationId) return;
    window.history.pushState({}, '', `/?conversation=${conversationId}`);
    const event = new CustomEvent('popstate');
    window.dispatchEvent(event);
  };

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      <Sidebar 
        selectedConversationId={selectedConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={onNewChat}
      />
      <div className="flex flex-col flex-1 h-full overflow-hidden">
        <div className="flex-1 overflow-hidden">
          <ChatMessages messages={messages} />
        </div>
        <div className="border-t bg-background p-4">
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
} 