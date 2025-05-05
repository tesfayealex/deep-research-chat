'use client';
import { ChatLayout } from "@/components/chat/chat-layout";
import React, { useEffect } from 'react';

export default function Home() {
  const [selectedConversationId, setSelectedConversationId] = React.useState<string | null>(null);
  
  // Check URL for conversation parameter on initial load and popstate events
  useEffect(() => {
    const handleConversationFromURL = () => {
      // Read conversation ID from URL if present
      const urlParams = new URLSearchParams(window.location.search);
      const conversationId = urlParams.get('conversation');
      setSelectedConversationId(conversationId);
    };
    
    // Parse URL on initial load
    handleConversationFromURL();
    
    // Listen for navigation events (back/forward buttons or history.pushState)
    window.addEventListener('popstate', handleConversationFromURL);
    
    return () => {
      window.removeEventListener('popstate', handleConversationFromURL);
    };
  }, []);
  
  const onNewChat = () => {
    // Clear the conversation ID
    setSelectedConversationId(null);
    
    // Update the URL to remove the conversation parameter
    window.history.pushState({}, '', '/');
  };
  
  return (
    <main className="min-h-screen">
      <ChatLayout selectedConversationId={selectedConversationId} onNewChat={onNewChat} />
    </main>
  );
}
