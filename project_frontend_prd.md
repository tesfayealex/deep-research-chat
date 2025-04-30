# PRD: Deep Research Agent - Frontend

## 1. Overview

The frontend will provide a user-friendly chat interface for interacting with the Deep Research Agent. It should feel responsive, modern, and clearly display both user inputs and the agent's outputs, including intermediate reasoning steps if possible. Inspiration is drawn heavily from the Vercel AI Chatbot.

## 2. Key Components & Features

1.  **Chat Container:**
    *   Displays the conversation history between the user and the agent.
    *   Distinguishes clearly between user messages and agent messages (e.g., different alignment or background colors).
    *   Should handle scrolling smoothly as the conversation grows.

2.  **Message Display:**
    *   Renders user prompts simply.
    *   Renders agent responses, potentially using Markdown for formatting (lists, bolding, code blocks, links).
    *   **Advanced:** Include a mechanism to optionally show the agent's internal "thought process" or the steps it's taking (Plan, Execute: Searching 'X', Process: Analyzing results, Refine: Modifying query, Report: Compiling findings). This could be a collapsible section within the agent's message bubble or a separate status indicator area.

3.  **Input Bar:**
    *   A text area for the user to type their research requests or follow-up questions.
    *   A "Send" button or ability to submit by pressing Enter.
    *   Potentially include buttons for common actions like "Start New Research" or "Stop Current Task".
    *   Loading indicator while the agent is processing the request.

4.  **Styling & Responsiveness:**
    *   Clean, modern aesthetic similar to `chat.vercel.ai`.
    *   Responsive design that works well on both desktop and mobile browsers.

## 3. Technology Stack (Recommendation)

*   **Framework:** React or Next.js (aligns with `vercel/ai-chatbot` example)
*   **Styling:** Tailwind CSS (commonly used with Next.js, good for rapid UI development) or CSS Modules.
*   **State Management:** React Context API, Zustand, or Redux Toolkit (choose based on complexity).
*   **Data Fetching:** `fetch` API, `axios`, or a library like `react-query` / `swr` for interacting with the backend API.

## 4. Sample Code Snippets (Conceptual)

These are illustrative examples of component structure, not complete implementations.

**Chat Interface Component (React):**

```jsx name=ChatInterface.jsx
import React, { useState, useEffect } from 'react';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import AgentStatus from './AgentStatus'; // Optional component for reasoning steps

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [agentStatus, setAgentStatus] = useState(''); // e.g., "Planning...", "Searching...", "Idle"
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (inputText) => {
    setIsLoading(true);
    setMessages(prev => [...prev, { sender: 'user', text: inputText }]);
    setAgentStatus('Processing...'); // Initial status

    try {
      // --- API Call to Backend ---
      const response = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inputText }),
      });
      const data = await response.json(); // Assuming backend sends back the final report

      // TODO: Handle streaming responses or intermediate status updates if implemented
      setMessages(prev => [...prev, { sender: 'agent', text: data.report }]);
      setAgentStatus('Idle');

    } catch (error) {
      console.error("Error contacting backend:", error);
      setMessages(prev => [...prev, { sender: 'agent', text: 'Sorry, I encountered an error.' }]);
      setAgentStatus('Error');
    } finally {
      setIsLoading(false);
    }
  };

  // Effect to potentially listen for status updates via WebSocket or SSE
  useEffect(() => {
    // Setup listener for agent status updates from backend (if implemented)
    // updateAgentStatus(newStatus);
  }, []);


  return (
    <div className="chat-container">
      {/* Optional: Display agent's current high-level task */}
      {/* <AgentStatus status={agentStatus} /> */}

      <MessageList messages={messages} />
      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  );
}

export default ChatInterface;
```

**Message Display (Conceptual):**

```jsx name=Message.jsx
function Message({ message }) {
  const isAgent = message.sender === 'agent';
  const messageClass = isAgent ? 'message agent-message' : 'message user-message';

  return (
    <div className={messageClass}>
      {/* Render message text, potentially using a Markdown renderer for agent messages */}
      <p>{message.text}</p>
      {/* Optional: Add timestamp or agent reasoning steps here */}
    </div>
  );
}
```